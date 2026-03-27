import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from einops import repeat
from functools import partial

from models.utils import *
from utils import new_utils


class MaskTransformer(nn.Module):
    def __init__(self, 
                 code_dim, 
                 cond_mode, 
                 latent_dim=256, 
                 ff_size=1024, 
                 num_layers=8,
                 num_heads=4, 
                 dropout=0.1, 
                 clip_dim=512, 
                 cond_drop_prob=0.1,
                 args=None, 
                 **kargs):
        super(MaskTransformer, self).__init__()

        # parameters
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        
        self.dropout = dropout
        
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        self.num_actions = kargs.get('num_actions', 1)

        self.args = args

        # modules
        if self.cond_mode == 'text':
            self.text_feat_dim = self.clip_dim
            self.cond_emb = nn.Linear(self.text_feat_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!")

        self.input_process = InputProcess(self.code_dim, self.latent_dim)

        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.num_heads = num_heads
        
        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        nb_token = args.nb_code + 2
        self.mask_id = args.nb_code
        self.pad_id = args.nb_code + 1

        self.token_emb = nn.Embedding(nb_token, self.code_dim)
        
        self.output_process = OutputProcess_Bert(out_feats=args.nb_code, latent_dim=self.latent_dim)

        self.segment_align_block = SegmentAlignBlock(emb_dim=self.latent_dim,
                                                     temp=args.align_temp)

        self.apply(self.init_weights)
        
        self.noise_schedule = cosine_schedule
    
# ************************************************************
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def load_and_freeze_token_emb(self, codebook):
        assert self.training, 'Only necessary in training mode'

        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) 
        self.token_emb.requires_grad_(False)
        print("Token embedding initialized!")

    def encode_text(self, text):
        feat_text = text
        
        return feat_text

    def encode_cond(self, cond):
        bs = len(cond)
        device = next(self.parameters()).device
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(cond)
        elif self.cond_mode == 'action':
            cond_vector = self.encode_action(cond).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!")
        
        return cond_vector

    def mask_cond(self, cond, force_mask=False):
        # randomly mask (all-zero) the condition vector of some samples in the batch using 'self.cond_drop_prob'
        bs =  cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            #* action embedding
            if cond.ndim == 3:
                mask = mask.view(bs, 1, 1)
            return cond * (1. - mask)
        else:
            return cond
# ************************************************************

    def trans_forward(self, 
                      motion_ids, 
                      cond, 
                      padding_mask, 
                      action_mask,
                      motion_seg,
                      force_mask=False):
        cond = self.mask_cond(cond, force_mask=force_mask)

        cond = self.cond_emb(cond).permute(1, 0, 2)

        x = self.token_emb(motion_ids)
        x = self.input_process(x)
        x = self.position_enc(x)

        xseq = torch.cat([cond, x], dim=0)

        # sample: [False, False, False, ..., True, True]
        cond_num = cond.shape[0]
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), 
                                  ~action_mask, 
                                  padding_mask], dim=1)

        # Only select the last seq_len tokens
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)
        
        action_emb = output[1:cond_num].permute(1, 0, 2)

        motion_output = output[cond_num:]
        motion_emb = motion_output.permute(1, 0, 2)
        motion_mask = ~padding_mask[:, cond_num:]
        
        align_loss = self.segment_align_block(
            action_emb, motion_emb, action_mask, motion_mask, motion_seg)
        
        logits = self.output_process(motion_output)

        return_dict = {
            'logits': logits,
            'align_loss': align_loss
        }

        return return_dict

    def forward(self, 
                motion_ids, 
                cond, 
                m_lens, 
                action_mask,
                motion_seg):
        bs, seq_len = motion_ids.shape
        device = motion_ids.device

        # nothing to do
        cond_vector = self.encode_cond(cond)
        force_mask = True if self.cond_mode == 'uncond' else False

        #! (1) Pad the motion according to real length
        # Positions that are PADDED are set as FALSE
        non_pad_mask = lengths_to_mask(m_lens, seq_len)
        motion_ids = torch.where(non_pad_mask, motion_ids, self.pad_id)

        #! (2) Generate a random mask
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((bs, seq_len), device=device).argsort(dim=-1)
        # Positions to be MASKED are set as TRUE
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        # Positions to be MASKED must also be NON-PADDED
        mask &= non_pad_mask
        # 'labels' is the training target, not input
        labels = torch.where(mask, motion_ids, self.mask_id)
        masked_motion_ids = motion_ids.clone()

        #! (3) Apply BERT masking
        # Step 1: 10% replace with incorrect/random token
        mask_rand_id = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(masked_motion_ids, high=self.args.nb_code)
        masked_motion_ids = torch.where(mask_rand_id, rand_id, masked_motion_ids)
        # Step 2: 90% x 88% replace with mask token
        mask_mask_id = get_mask_subset_prob(mask & ~mask_rand_id, 0.88)
        masked_motion_ids = torch.where(mask_mask_id, self.mask_id, masked_motion_ids)

        return_dict = self.trans_forward(
            masked_motion_ids, cond_vector, ~non_pad_mask, action_mask, motion_seg, force_mask=force_mask)
        ce_loss, pred_id, acc = cal_performance(
            return_dict['logits'], labels, ignore_index=self.mask_id)

        return_dict['pred_ids'] = pred_id
        return_dict['ce_loss'] = ce_loss
        return_dict['acc'] = acc

        return return_dict

    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                action_mask,
                                motion_seg,
                                cond_scale=3,
                                force_mask=False):
        if force_mask:
            return_dict = self.trans_forward(
                motion_ids, cond_vector, padding_mask, action_mask, motion_seg, force_mask=True)
            return return_dict

        return_dict = self.trans_forward(
            motion_ids, cond_vector, padding_mask, action_mask, motion_seg)
        if cond_scale == 1:
            return return_dict

        aux_return_dict = self.trans_forward(
            motion_ids, cond_vector, padding_mask, action_mask, motion_seg, force_mask=True)

        # update logits
        logits = return_dict['logits']
        aux_logits = aux_return_dict['logits']
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return_dict['logits'] = scaled_logits
        
        return return_dict

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 cond,
                 m_lens,
                 timesteps,
                 cond_scale,
                 action_mask,
                 motion_seg,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False):
        # not always 49
        seq_len = max(m_lens)
        device = m_lens.device
        
        # nothing to do
        cond_vector = self.encode_cond(cond)
        
        # (1) Start from all tokens (except padding) being masked
        # Positions that are PADDED are set as TRUE
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        # generate the in 'timesteps' steps
        for timestep, _ in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = self.noise_schedule(timestep)
            
            # (2) Select num_token_masked tokens with lowest scores to be masked
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            sorted_indices = scores.argsort(dim=1) 
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            motion_ids = torch.where(is_mask, self.mask_id, motion_ids)

            # (3) Predict the complete motion_ids according to the masked motion_ids
            return_dict = self.forward_with_cond_scale(
                motion_ids, cond_vector, padding_mask, action_mask, motion_seg, cond_scale=cond_scale, force_mask=force_mask)
            
            logits = return_dict['logits'].permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            temperature = starting_temperature
            if gsample:  
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else: 
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                pred_ids = Categorical(probs).sample()
            motion_ids = torch.where(is_mask, pred_ids, motion_ids)

            # (4) Update scores (for next iteration)
            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))
            scores = scores.squeeze(-1)
            # do not re-mask the previously kept tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        motion_ids = torch.where(padding_mask, -1, motion_ids)

        return motion_ids


class ResidualTransformer(nn.Module):
    def __init__(self, 
                 code_dim, 
                 cond_mode, 
                 latent_dim=256, 
                 ff_size=1024, 
                 num_layers=8, 
                 cond_drop_prob=0.1,
                 num_heads=4, 
                 dropout=0.1, 
                 clip_dim=512, 
                 shared_codebook=False, 
                 share_weight=False,
                 args=None, 
                 **kargs):
        super(ResidualTransformer, self).__init__()

        # parameters
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim

        self.dropout = dropout
        
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        
        self.shared_codebook = shared_codebook
        self.share_weight = share_weight

        self.num_actions = kargs.get('num_actions', 1)

        self.args = args

        # modules
        if self.cond_mode == 'text':
            self.text_feat_dim = self.clip_dim
            self.cond_emb = nn.Linear(self.text_feat_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        else:
            raise KeyError("Unsupported condition mode!")

        self.input_process = InputProcess(self.code_dim, self.latent_dim)

        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=num_heads,
                                                            dim_feedforward=ff_size,
                                                            dropout=dropout,
                                                            activation='gelu')
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=num_layers)

        self.num_heads = num_heads
        
        self.encode_quant = partial(F.one_hot, num_classes=self.args.num_quantizers)

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        self.quant_emb = nn.Linear(self.args.num_quantizers, self.latent_dim)

        nb_token = args.nb_code + 1 
        self.pad_id = args.nb_code

        self.output_process = OutputProcess(out_feats=code_dim, latent_dim=latent_dim)

        if shared_codebook:
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(nb_token, code_dim)))
            self.token_embed_weight = token_embed.expand(args.num_quantizers-1, nb_token, code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(nb_token, code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(nb_token,)))
                self.output_proj_weight = output_proj.expand(args.num_quantizers-1, nb_token, code_dim)
                self.output_proj_bias = output_bias.expand(args.num_quantizers-1, nb_token)
        else:
            if share_weight:
                self.embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(args.num_quantizers - 2, nb_token, code_dim)))

                self.token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, nb_token, code_dim)))
                
                self.output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, nb_token, code_dim)))
                self.output_proj_bias = None
                
                self.registered = False
            else:
                token_embed_weight = torch.normal(mean=0, std=0.02, size=(args.num_quantizers - 1, nb_token, code_dim))
                self.token_embed_weight = nn.Parameter(token_embed_weight)

                output_proj_weight = torch.normal(mean=0, std=0.02, size=(args.num_quantizers - 1, nb_token, code_dim))
                self.output_proj_weight = nn.Parameter(output_proj_weight)
                self.output_proj_bias = nn.Parameter(torch.zeros(size=(args.num_quantizers, nb_token)))

        self.apply(self.init_weights)

# ************************************************************
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode_text(self, text):
        feat_text = text

        return feat_text

    def encode_cond(self, cond):
        bs = len(cond)
        device = next(self.parameters()).device
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(cond)
        elif self.cond_mode == 'action':
            cond_vector = self.encode_action(cond).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!")
    
        return cond_vector

    def process_embed_proj_weight(self):
        if self.share_weight and (not self.shared_codebook):
            self.token_embed_weight = torch.cat([self.token_embed_weight_, self.embed_proj_shared_weight], dim=0)
            self.output_proj_weight = torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)

    def output_project(self, logits, qids):
        '''
        :logits: (bs, code_dim, seqlen)
        :qids: (bs,)
        :return:
            -logits (bs, ntoken, seqlen)
        '''

        # (num_qlayers-1, num_token, code_dim) -> (bs, ntoken, code_dim)
        output_proj_weight = self.output_proj_weight[qids]
        # (num_qlayers, ntoken) -> (bs, ntoken)
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]

        output = torch.einsum('bnc, bcs->bns', output_proj_weight, logits)
        if output_proj_bias is not None:
            output += output + output_proj_bias.unsqueeze(-1)
        return output

    def mask_cond(self, cond, force_mask=False):
        bs =  cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            if cond.ndim == 3:
                mask = mask.view(bs, 1, 1)
            return cond * (1. - mask)
        else:
            return cond
# ************************************************************

    def trans_forward(self, 
                      motion_codes, 
                      qids, 
                      cond, 
                      padding_mask, 
                      action_mask,
                      force_mask=False):
        cond = self.mask_cond(cond, force_mask=force_mask)
        
        cond = self.cond_emb(cond).permute(1, 0, 2)

        x = self.input_process(motion_codes)
        x = self.position_enc(x)

        q_onehot = self.encode_quant(qids).float().to(x.device)
        q_emb = self.quant_emb(q_onehot).unsqueeze(0)

        xseq = torch.cat([cond, q_emb, x], dim=0)

        cond_num = cond.shape[0] + 1
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), 
                                  ~action_mask, 
                                  torch.zeros_like(padding_mask[:, 0:1]),
                                  padding_mask], dim=1)
    
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)

        logits = self.output_process(output[cond_num:])

        return logits

    def forward(self, 
                all_ids, 
                cond, 
                m_lens, 
                action_mask):
        bs, seq_len, num_quant_layers = all_ids.shape
        device = all_ids.device

        cond_vector = self.encode_cond(cond)
        force_mask = True if self.cond_mode == 'uncond' else False
        
        self.process_embed_proj_weight()

        non_pad_mask = lengths_to_mask(m_lens, seq_len)
        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_ids = torch.where(q_non_pad_mask, all_ids, self.pad_id)

        # Randomly select quantization layers to work on
        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

        token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
        gather_indices = repeat(all_ids[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
        all_codes = token_embed.gather(1, gather_indices)
        cumsum_codes = torch.cumsum(all_codes, dim=-1)

        active_indices = all_ids[torch.arange(bs), :, active_q_layers]
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]

        logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask, action_mask, force_mask)
        logits = self.output_project(logits, active_q_layers-1)

        ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)

        return ce_loss, pred_id, acc, active_q_layers
    
    def forward_with_cond_scale(self,
                                motion_codes,
                                q_id,
                                cond_vector,
                                padding_mask,
                                action_mask,
                                cond_scale=3,
                                force_mask=False):
        bs = motion_codes.shape[0]

        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
        if force_mask:
            logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, action_mask, force_mask=True)
            logits = self.output_project(logits, qids-1)
            return logits

        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, action_mask)
        logits = self.output_project(logits, qids-1)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, action_mask, force_mask=True)
        aux_logits = self.output_project(aux_logits, qids-1)
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 motion_ids,
                 cond,
                 m_lens,
                 action_mask,
                 temperature=1,
                 topk_filter_thres=0.9,
                 cond_scale=2,
                 num_res_layers=-1):
        bs, seq_len = motion_ids.shape
        
        cond_vector = self.encode_cond(cond)
        
        self.process_embed_proj_weight()
        
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_ids = [motion_ids]
        history_sum = 0
        num_quant_layers = self.args.num_quantizers if num_res_layers==-1 else num_res_layers+1

        # generate each layer step by step, each layer in one step
        for i in range(1, num_quant_layers):
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=bs)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, action_mask, cond_scale=cond_scale)
            logits = logits.permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            next_ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = next_ids
            all_ids.append(next_ids)

        all_ids = torch.stack(all_ids, dim=-1)
        all_ids = torch.where(all_ids==self.pad_id, -1, all_ids)
        return all_ids


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute((1, 0, 2)) # [seqlen, bs, input_feats]
        x = self.poseEmbedding(x)  # [seqlen, bs, input_feats]
        return x


class PositionalEncoding(nn.Module):
    # Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)
        output = output.permute(1, 2, 0)
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states) 
        output = output.permute(1, 2, 0)
        return output


class SegmentAlignBlock(nn.Module):
    def __init__(self, emb_dim=384, temp=0.1):
        super().__init__()
        
        self.temp = temp

        self.gather_ln = nn.Linear(emb_dim*2, emb_dim)

    def gather_mean(self, motion_seg_emb, motion_seg_mask):
        motion_seg_emb_sum = (motion_seg_emb * motion_seg_mask.unsqueeze(-1)).sum(dim=2)  # [B, A, D]
        seg_lengths = motion_seg_mask.sum(dim=2).clamp(min=1)  # [B, A]
        motion_seg_emb_mean = motion_seg_emb_sum / seg_lengths.unsqueeze(-1) 
        return motion_seg_emb_mean
    
    def gather_max(self, motion_seg_emb, motion_seg_mask):
        motion_seg_emb_masked = torch.where(
            motion_seg_mask.unsqueeze(-1).bool(),
            motion_seg_emb,
            torch.full_like(motion_seg_emb, float('-1e9'))
        )  # [B, A, T, D]
        motion_seg_emb_max = motion_seg_emb_masked.max(dim=2).values  # [B, A, D]
        seg_lengths = motion_seg_mask.sum(dim=2).clamp(max=1)  # [B, A]
        motion_seg_emb_max = motion_seg_emb_max * seg_lengths.unsqueeze(-1)
        return motion_seg_emb_max

    def get_align_loss(self, text_seg_emb, motion_seg_emb):
        B, T, D = text_seg_emb.shape

        a = F.normalize(text_seg_emb, dim=-1)
        m = F.normalize(motion_seg_emb, dim=-1)

        sim_a2m = torch.matmul(a, m.transpose(1, 2))
        sim_m2a = sim_a2m.transpose(1, 2)

        sim_a2m_scaled = sim_a2m / self.temp
        sim_m2a_scaled = sim_m2a / self.temp

        target = torch.arange(T).to(sim_a2m_scaled.device).repeat(B)
            
        loss_a2m = F.cross_entropy(sim_a2m_scaled.reshape(B*T, T), target)
        loss_m2a = F.cross_entropy(sim_m2a_scaled.reshape(B*T, T), target)

        loss = (loss_a2m + loss_m2a) / 2

        return loss

    def forward(self, action_emb, motion_emb, action_mask, motion_mask, motion_seg):
        start_idxs, end_idxs = new_utils.get_seg_idxs(motion_mask, action_mask, motion_seg)
        motion_seg_emb, motion_seg_mask = new_utils.get_segs_from_idxs(motion_emb, start_idxs, end_idxs)

        # print(motion_seg_emb.shape)  # [B, A, T, D]
        # print(motion_seg_mask.shape)  # [B, A, T]

        motion_seg_emb_mean = self.gather_mean(motion_seg_emb, motion_seg_mask)
        motion_seg_emb_max = self.gather_max(motion_seg_emb, motion_seg_mask)
        motion_seg_emb_concat = torch.cat([motion_seg_emb_mean, motion_seg_emb_max], dim=-1)
        motion_seg_emb = self.gather_ln(motion_seg_emb_concat)
        
        # print(motion_seg_emb.shape)  # [B, A, D]
            
        align_loss = self.get_align_loss(action_emb, motion_seg_emb)

        return align_loss
    