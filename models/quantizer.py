import numpy as np
from einops import rearrange
import random
from random import randrange
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature=1.,
    stochastic=False,
    dim=-1,
    training=True
):
    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)

    return ind


class QuantizerEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super(QuantizerEMAReset, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.):
        k_w = self.codebook.t()

        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - \
                   2 * torch.matmul(x, k_w) + \
                   torch.sum(k_w ** 2, dim=0, keepdim=True)  # (N * L, b)

        code_idx = gumbel_sample(-distance, dim=-1, temperature=sample_codebook_temp, stochastic=True, training=self.training)

        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * code_rand

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(-1, x.shape[-1])
        x = rearrange(x, 'n c t -> (n t) c')
        return x

    def forward(self, x, return_idx=False, temperature=0.):
        N, width, T = x.shape

        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach()) # It's right. the t2m-gpt paper is wrong on embed loss and commitment loss.

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()
        # print(code_idx[0])
        if return_idx:
            return x_d, code_idx, commit_loss, perplexity
        return x_d, commit_loss, perplexity


class QuantizerEMA(QuantizerEMAReset):
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * self.codebook

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity


class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        if shared_codebook:
            layer = QuantizerEMAReset(**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizerEMAReset(**kwargs) for _ in range(num_quantizers)])

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_prob = quantize_dropout_prob
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
  
    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks
    
    def get_codes_from_indices(self, indices):
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct
        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        # get ready for gathering
        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b=batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d=codebooks.shape[-1])

        # take care of quantizer dropout
        mask = gather_indices == -1.
        # have it fetch a dummy code to be masked out later
        gather_indices = gather_indices.masked_fill(mask, 0) 

        # gather all codes
        all_codes = codebooks.gather(2, gather_indices)

        # mask out any codes that were dropout
        all_codes = all_codes.masked_fill(mask, 0.)

        return all_codes # [q, b, n, d]

    def get_codebook_entry(self, indices): # [b, n, q]
        all_codes = self.get_codes_from_indices(indices) # [q, b, n, d]
        latent = torch.sum(all_codes, dim=0) # [b, n, d]
        latent = latent.permute(0, 2, 1) # [b, d, n]
        return latent

    def forward(self, x, return_all_codes=False, sample_codebook_temp=None, force_dropout_index=-1):
        num_quant = self.num_quantizers
        device = x.device

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_perplexity = []

        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob
        # using no dropout by default
        start_drop_quantize_index = num_quant
        
        # To ensure the first-k layers learn things as much as possible, we randomly dropout the last q - k layers
        if should_quantize_dropout:
            # keep quant layers <= quantize_dropout_cutoff_index, TODO vary in batch
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant) 
            null_indices_shape = [x.shape[0], x.shape[-1]]
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

        # set dropout index by force
        if force_dropout_index >= 0:
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index
            null_indices_shape = [x.shape[0], x.shape[-1]]
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

        for quantizer_index, layer in enumerate(self.layers):
            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices)
                continue

            quantized, *rest = layer(residual, return_idx=True, temperature=sample_codebook_temp)
            residual -= quantized.detach()
            quantized_out += quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)

        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses)/len(all_losses)
        all_perplexity = sum(all_perplexity)/len(all_perplexity)
        ret = (quantized_out, all_indices, all_losses, all_perplexity)

        if return_all_codes:
            all_codes = self.get_codes_from_indices(all_indices)
            ret = (*ret, all_codes)

        return ret
    
    def quantize(self, x, return_latent=False):
        all_indices = []
        quantized_out = 0.
        residual = x
        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):
            quantized, *rest = layer(residual, return_idx=True)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx
