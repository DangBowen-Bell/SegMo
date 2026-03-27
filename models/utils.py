import torch
import torch.nn.functional as F
import math
from einops import rearrange
import time
import numpy as np


################################################## 
# return mask where padding is FALSE
def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# return mask where padding is ALL FALSE
def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)


# Given seq: (b, s)
# Return mat: (1, s, s)
# Example Output:
#        [[[ True, False, False], k
#          [ True,  True, False],
#          [ True,  True,  True]]]
# For causal attention
def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

################################################## tensor helpers
# Get a random subset of TRUE mask, with prob
def get_mask_subset_prob(mask, prob):
    # for torch.bernoulli() function, the first parameter only determines the output shape
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


# Get mask of special_tokens in ids
def get_mask_special_tokens(ids, special_ids):
    mask = torch.zeros_like(ids).bool()
    for special_id in special_ids:
        mask |= (ids==special_id)
    return mask

# network builder helpers
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


################################################## classifier free guidance functions
def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


################################################## sampling helpers
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = 1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

# Example input:
#        [[ 0.3596,  0.0862,  0.9771, -1.0000, -1.0000, -1.0000],
#         [ 0.4141,  0.1781,  0.6628,  0.5721, -1.0000, -1.0000],
#         [ 0.9428,  0.3586,  0.1659,  0.8172,  0.9273, -1.0000]]
# Example output:
#        [[  -inf,   -inf, 0.9771,   -inf,   -inf,   -inf],
#         [  -inf,   -inf, 0.6628,   -inf,   -inf,   -inf],
#         [0.9428,   -inf,   -inf,   -inf,   -inf,   -inf]]
def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    return probs


################################################## noise schedules
# More on large value, less on small
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def scale_cosine_schedule(t, scale):
    return torch.clip(scale*torch.cos(t * math.pi * 0.5) + 1 - scale, min=0., max=1.)

# More on small value, less on large
def q_schedule(bs, low, high, device):
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low


def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc


def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss


def cal_patch_weights(v_patch_nums):
    alpha = 1.0 
    raw_weights = 1 / (np.array(v_patch_nums) ** alpha)
    normalized_weights = raw_weights / raw_weights.sum()
    patch_weights = np.concatenate([np.full(l, w) for l, w in zip(v_patch_nums, normalized_weights)])
    patch_weights *= (len(patch_weights) / patch_weights.sum())
    return patch_weights


def cal_performance_weighted(pred, labels, ignore_index=None, smoothing=0., tk=1, patch_weights=None):
    loss = cal_loss_weighted(pred, labels, ignore_index, smoothing=smoothing, patch_weights=patch_weights)
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc


def cal_loss_weighted(pred, labels, ignore_index=None, smoothing=0., patch_weights=None):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''

    patch_weights = torch.tensor(patch_weights, dtype=pred.dtype, device=pred.device)

    if smoothing:
        # NO WEIGHTED LOSS IMPLEMENTATION
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        # loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index, reduction='none')
        loss = (loss * patch_weights).mean()

    return loss


def cal_performance_by_patch(pred, labels, ignore_index=None, smoothing=0., tk=1, v_patch_nums=None):
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)

    #! calculate accuracy/loss for each patch
    # B=64, k=1, L=206
    # pred_id_k: [B, k, L]
    # labels: [B, L]
    # mask: [B, L]
    acc_per_patch = []
    start_i = 0
    for pi, pn in enumerate(v_patch_nums):
        n_correct_patch = (pred_id_k[:, :, start_i:start_i+pn] == labels[:, start_i:start_i+pn].unsqueeze(1)).any(dim=1).masked_select(mask[:, start_i:start_i+pn])
        # print(n_correct_patch.shape)
        acc_patch = torch.mean(n_correct_patch.float()).item()
        acc_per_patch.append(acc_patch)
        start_i += pn
    # print(acc_per_patch)
    
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc, acc_per_patch


################################################## other
def def_value():
    return 0.0


def print_current_loss(start_time, iter, max_iter, losses, 
                       epoch=None, inner_iter=None):
    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        print('ep/it:%2d-%4d niter:%6d' % (epoch, inner_iter, iter), end=" ")

    message = ' %s completed:%3d%%)' % (time_since(start_time, iter / max_iter), iter / max_iter * 100)

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)

    print(message)