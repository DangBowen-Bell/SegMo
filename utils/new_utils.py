import torch
import datetime
import os
import os.path as osp
from os.path import join as pjoin
import numpy as np
import codecs as cs
import random
import logging
import sys
from argparse import Namespace
import re
from transformers import AutoModel, AutoTokenizer

from models.word_vectorizer import WordVectorizer
from models.evaluator import EvaluatorModelWrapper
from models.rvqvae import RVQVAE
from models.utils import lengths_to_mask
from models.trans import MaskTransformer, ResidualTransformer

from options import option_mtrans, option_rtrans


############################################################


data_root = '/root/dir/of/dataset/'


############################################################


class DatasetInfo(object):
    def __init__(self, dataname, usage='train'):
        if dataname == 't2m':
            self.data_root = pjoin(data_root, 'HumanML3D')
            self.joints_num = 22
            self.dim_pose = 263
            self.min_motion_length = 40
            self.fps = 20
        elif dataname == 'kit':
            self.data_root = pjoin(data_root, 'KIT-ML')    
            self.joints_num = 21
            self.dim_pose = 251
            self.min_motion_length = 24
            self.fps = 12.5
    
        self.max_motion_length = 196    
        
        self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        self.text_dir = pjoin(self.data_root, 'texts')
        
        meta_dir = f'checkpoints/{dataname}/Comp_v6_KLD005/meta'
        mean_path = pjoin(meta_dir, 'mean.npy')
        std_path = pjoin(meta_dir, 'std.npy')

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        split_file = pjoin(self.data_root, f'{usage}.txt')
        self.id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                self.id_list.append(line.strip())


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_output_dir(args):
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.out_dir = f"./{args.out_dir}_{args.dataname}/{date}_{args.exp_name}/"
    os.makedirs(args.out_dir, exist_ok=False)


def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def load_word_vectorizer():
    w_vectorizer = WordVectorizer('glove', 'our_vab')

    return w_vectorizer


def load_eval_wrapper(dataname):
    dataset_opt_path = f'checkpoints/{dataname}/Comp_v6_KLD005/opt.txt'
    eval_opt = load_config_eval(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(eval_opt)
    
    return eval_wrapper


def load_text_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.clip_version)
    model = AutoModel.from_pretrained(args.clip_version).to(args.device)
    
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return tokenizer, model


def encode_text_hf(text, text_model):
    tokenizer, model = text_model
    max_length = tokenizer.model_max_length

    device = model.device

    inputs = tokenizer(text, 
                       padding='max_length', 
                       max_length=max_length,
                       truncation=True, 
                       return_tensors='pt').to(device)
    
    input_ids = inputs.input_ids
    if input_ids.shape[-1] > max_length:
        input_ids = input_ids[:, :max_length]

    text_emb = model.get_text_features(input_ids.to(device))
    text_emb = text_emb.unsqueeze(1)

    word_emb = model.text_model(input_ids.to(device)).last_hidden_state

    return text_emb, word_emb


def get_action_emb(cond, llm_captions, text_model, max_action_num=5):
    B = len(cond)
    N = max_action_num + 1

    action_num = [len(caps) for caps in llm_captions]
    text_list = []
    for bi, (prefix, caps) in enumerate(zip(cond, llm_captions)):
        text_list += [prefix]
        text_list += caps
        text_list += [''] * (max_action_num - len(caps))

    new_cond, _ = encode_text_hf(text_list, text_model)
    new_cond = new_cond.squeeze(1).view(B, N, -1)

    action_num = torch.tensor(action_num)
    action_mask = lengths_to_mask(action_num, max_action_num)

    action_mask = action_mask.to(new_cond.device)

    return new_cond, action_mask


def get_seg_idxs(motion_mask, action_mask, motion_seg):
    # print('action_mask: ', action_mask[:5])
    
    B, T = motion_mask.shape
    _, A = action_mask.shape
    device = motion_mask.device
    
    motion_lengths = motion_mask.sum(dim=-1)
    # print('motion_lengths: ', motion_lengths[:5])

    seg_lengths_masked = motion_seg.detach().clone().to(device)
    # print('seg_lengths_masked: ', seg_lengths_masked[:5])

    end_idxs = torch.cumsum(seg_lengths_masked, dim=1) - 1
    start_idxs = torch.zeros_like(end_idxs)
    motion_lengths_exp = motion_lengths.unsqueeze(-1).expand(-1, A - 1)
    start_idxs[:, 1:] = torch.minimum(end_idxs[:, :-1] + 1, motion_lengths_exp - 1)
    # print('start_idxs: ', start_idxs[:5])
    # print('end_idxs: ', end_idxs[:5])
    
    return start_idxs, end_idxs


def get_segs_from_idxs(motion_emb, start_idxs, end_idxs):
    if motion_emb.dim() == 3:
        _, T, D = motion_emb.shape
    else:
        _, T = motion_emb.shape

    _, A = start_idxs.shape
    device = motion_emb.device

    start_idxs = start_idxs.long()
    end_idxs = end_idxs.long()

    offsets = torch.arange(T, device=device).view(1, 1, -1)  # [1, 1, T]
    idxs = start_idxs.unsqueeze(-1) + offsets  # [B, A, T]
    clamped_idxs = torch.clamp(idxs, max=T-1)
    # print('offsets: ', offsets[:2])
    # print('idxs: ', idxs[:2])
    # print('clamped_idxs: ', clamped_idxs[:2])

    if motion_emb.dim() == 3:
        motion_emb_exp = motion_emb.unsqueeze(1).expand(-1, A, -1, -1)  # [B, A, T, D]
        clamped_idxs = clamped_idxs.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, A, T, D]
        seg_emb = torch.gather(motion_emb_exp, dim=2, index=clamped_idxs)  # [B, A, T, D]
    else:
        motion_emb_exp = motion_emb.unsqueeze(1).expand(-1, A, -1)  # [B, A, T]
        seg_emb = torch.gather(motion_emb_exp, dim=2, index=clamped_idxs)  # [B, A, T]
    
    seg_lengths = (end_idxs - start_idxs + 1).unsqueeze(-1)  # [B, A, 1]
    # invalid = (end_idxs <= start_idxs).unsqueeze(-1)  # [B, A, 1]
    # seg_lengths = seg_lengths.masked_fill(invalid, 0)
    seg_mask = offsets < seg_lengths  # [B, A, T]
    # print('seg_lengths: ', seg_lengths[:2])
    # print('seg_mask: ', seg_mask[:2])

    return seg_emb, seg_mask


############################################################


def load_rvqvae_old(args):
    ckpt_dir = './checkpoints'
    ckpt_name = 'rvq_nq6_dc512_nc512_noshare_qdp0.2'
    if args.dataname == 'kit':
        ckpt_name = ckpt_name + '_k'
    cfg_path = pjoin(ckpt_dir, args.dataname, ckpt_name, 'opt.txt')
    args_vq = load_config(cfg_path)
    
    args_vq.dataname = args_vq.dataset_name
    
    vq_model = RVQVAE(
        args_vq,
        args_vq.nb_code,
        args_vq.code_dim,
        args_vq.down_t,
        args_vq.stride_t,
        args_vq.width,
        args_vq.depth,
        args_vq.dilation_growth_rate,
        args_vq.vq_act,
        args_vq.vq_norm)
    ckpt_path = pjoin(ckpt_dir, args.dataname, ckpt_name, 'model/net_best_fid.tar')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if args.dataname == 'kit':
        vq_model.load_state_dict(ckpt['vq_model'])
    else:
        vq_model.load_state_dict(ckpt['net'])
    print(f'Loading RVQ-VAE Model {args.vq_name}')
    return vq_model, args_vq


def load_rvqvae(args):
    cfg_path = pjoin('output/vq_' + args.dataname, args.vq_name, 'config.txt')
    args_vq = load_config(cfg_path)
    
    vq_model = RVQVAE(
        args_vq,
        args_vq.nb_code,
        args_vq.code_dim,
        args_vq.down_t,
        args_vq.stride_t,
        args_vq.width,
        args_vq.depth,
        args_vq.dilation_growth_rate,
        args_vq.vq_act,
        args_vq.vq_norm)
    ckpt_path = pjoin('output/vq_' + args.dataname, args.vq_name, args.model_name_vq + '.tar')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    vq_model.load_state_dict(ckpt['vq_model'])
    print(f'Loading RVQ-VAE Model {args.vq_name}')
    return vq_model, args_vq


def load_mtrans(args, args_vq):    
    args_mtrans = option_mtrans.get_args_parser()
    args_mtrans.code_dim = args_vq.code_dim
    args_mtrans.nb_code = args_vq.nb_code

    t2m_trans = MaskTransformer(
        code_dim=args_mtrans.code_dim,
        cond_mode='text',
        latent_dim=args_mtrans.latent_dim,
        ff_size=args_mtrans.ff_size,
        num_layers=args_mtrans.nb_layer,
        num_heads=args_mtrans.nb_head,
        dropout=args_mtrans.dropout,
        clip_dim=args_mtrans.clip_dim,
        cond_drop_prob=args_mtrans.cond_drop_prob,
        clip_version=args_mtrans.clip_version,
        args=args_mtrans)
    
    ckpt_path = pjoin('output/mtrans_' + args.dataname, args.mtrans_name, args.model_name_mtrans + '.tar')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    t2m_trans.load_state_dict(ckpt['t2m_trans'], strict=False)
    print(f'Loading Mask Transformer Model: {args.mtrans_name}')
    
    return t2m_trans, args_mtrans


def load_rtrans(args, args_vq):
    args_rtrans = option_rtrans.get_args_parser()

    args_rtrans.code_dim = args_vq.code_dim
    args_rtrans.nb_code = args_vq.nb_code
    args_rtrans.down_t = args_vq.down_t
    args_rtrans.num_quantizers = args_vq.num_quantizers
    args_rtrans.shared_codebook = args_vq.shared_codebook

    res_trans = ResidualTransformer(
        code_dim=args_rtrans.code_dim,
        cond_mode='text',
        latent_dim=args_rtrans.latent_dim,
        ff_size=args_rtrans.ff_size,
        num_layers=args_rtrans.nb_layer,
        num_heads=args_rtrans.nb_head,
        dropout=args_rtrans.dropout,
        clip_dim=args_rtrans.clip_dim,
        cond_drop_prob=args_rtrans.cond_drop_prob,
        clip_version=args_rtrans.clip_version,
        shared_codebook=args_rtrans.shared_codebook, 
        share_weight=args_rtrans.share_weight,
        args=args_rtrans)
    
    ckpt_path = pjoin('output/rtrans_' + args.dataname, args.rtrans_name, args.model_name_rtrans + '.tar')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'res_trans' not in ckpt.keys():
        res_trans.load_state_dict(ckpt['all'], strict=False)
    else:
        res_trans.load_state_dict(ckpt['res_trans'], strict=False)
    print(f'Loading Residual Transformer Model: {args.rtrans_name}')
    
    return res_trans, args_rtrans


############################################################


def save_config(args):
    args_dict = vars(args)
    print('------------ Options -------------')
    for k, v in sorted(args_dict.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    cfg_path = osp.join(args.out_dir, 'config.txt')
    with open(cfg_path, 'wt') as f:
        f.write('------------ Options -------------\n')
        for k, v in sorted(args_dict.items()):
            f.write('%s: %s\n' % (str(k), str(v)))
        f.write('-------------- End ----------------\n')
        
        
def load_config(cfg_path):
    args = Namespace()
    args_dict = vars(args)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', cfg_path)
    with open(cfg_path, 'r') as f:
        for line in f:
            if line.strip() not in skip:
                key, value = line.strip('\n').split(': ')
                if value in ('True', 'False'):
                    args_dict[key] = (value == 'True')
                elif is_float(value):
                    args_dict[key] = float(value)
                elif is_number(value):
                    args_dict[key] = int(value)
                else:
                    args_dict[key] = str(value)

    return args


def load_config_eval(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    opt_dict['which_epoch'] = 'finest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 't2m':
        opt.data_root = pjoin(data_root, 'HumanML3D')
        opt.joints_num = 22
        opt.dim_pose = 263
    elif opt.dataset_name == 'kit':
        opt.data_root = pjoin(data_root, 'KIT-ML')
        opt.joints_num = 21
        opt.dim_pose = 251
    else:
        raise KeyError('Dataset not recognized')
    
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.max_motion_length = 196
    opt.max_motion_frame = 196
    opt.max_motion_token = 55

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def collate_fn_t2m(batch):
    notnone_batch = [b for b in batch if b is not None]
    notnone_batch.sort(key=lambda x: x[3], reverse=True)
    
    caption = [b[0] for b in notnone_batch]
    motion = torch.tensor(np.array([b[1] for b in notnone_batch]))
    m_length = torch.tensor(np.array([b[2] for b in notnone_batch]))
    llm_captions = [b[3] for b in notnone_batch]
    motion_seg = torch.tensor(np.array([b[4] for b in notnone_batch]))
    
    return caption, motion, m_length, llm_captions, motion_seg


def collate_fn_t2m_eval(batch):
    notnone_batch = [b for b in batch if b is not None]
    notnone_batch.sort(key=lambda x: x[3], reverse=True)

    word_embeddings = torch.tensor(np.array([b[0] for b in notnone_batch]))
    pos_one_hots = torch.tensor(np.array([b[1] for b in notnone_batch]))
    caption = [b[2] for b in notnone_batch]
    sent_len = torch.tensor(np.array([b[3] for b in notnone_batch]))
    motion = torch.tensor(np.array([b[4] for b in notnone_batch]))
    m_length = torch.tensor(np.array([b[5] for b in notnone_batch]))
    tokens = [b[6] for b in notnone_batch]
    llm_captions = [b[7] for b in notnone_batch]
    motion_seg = torch.tensor(np.array([b[8] for b in notnone_batch]))
    
    return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens, llm_captions, motion_seg
