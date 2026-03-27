import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
import numpy as np
from os.path import join as pjoin

from options.option_eval import get_args_parser
from dataset.dataset_t2m import Text2MotionDataset
from utils.eval_funcs import evaluation
from models.utils import *
from utils import new_utils


if __name__ == '__main__':
    args = get_args_parser()
    args.is_train = False

    new_utils.fixseed(args.seed)

    eval_wrapper = new_utils.load_eval_wrapper(args.dataname)

    #* vq model
    if args.vq_name == '':
        vq_model, args_vq = new_utils.load_rvqvae_old(args)
    else:
        vq_model, args_vq = new_utils.load_rvqvae(args)
    vq_model.eval()
    vq_model.to(args.device)

    #* mtrans
    if args.mtrans_name != '':
        t2m_trans, args_mtrans = new_utils.load_mtrans(args, args_vq)
        t2m_trans.eval()
        t2m_trans.to(args.device)
    else:
        t2m_trans = None

    #* rtrans
    if args.rtrans_name != '':
        res_trans, args_rtrans = new_utils.load_rtrans(args, args_vq)
        res_trans.eval()
        res_trans.to(args.device)
    else:
        res_trans = None

    #* output dir
    if args.mtrans_name == '':
        args.out_dir = pjoin('./output/eval/vq', args.dataname)
        model_type = 'vq_model'
    elif args.rtrans_name == '':
        args.out_dir = pjoin('./output/eval/mtrans', args.dataname)
        model_type = 't2m_trans'
    else:
        args.out_dir = pjoin('./output/eval/rtrans', args.dataname)
        model_type = 'all'
    os.makedirs(args.out_dir, exist_ok=True)
    new_utils.save_config(args)

    logger = new_utils.get_logger(args.out_dir)
    logger.info(f'Using VQ Model: {args.vq_name}')
    logger.info(f'Using Mask Transformer: {args.mtrans_name}')
    logger.info(f'Using Residual Transformer: {args.rtrans_name}')
    
    text_model = new_utils.load_text_model(args)
            
    #* dataset
    val_set = Text2MotionDataset(
        args.dataname, 
        True,
        usage='test',
        unit_length=2**args_vq.down_t)

    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size,
        num_workers=1, 
        shuffle=True,
        drop_last=True,
        collate_fn=new_utils.collate_fn_t2m_eval)

    #################################################################
    #! evaluation

    metrics = {
        'fid': [],
        'top1': [],
        'top2': [],
        'top3': [],
        'matching': [],
        'div': [],
        'mm': []
    }

    for i in range(args.repeat_time):
        with torch.no_grad():
            _metrics = evaluation(
                val_loader, eval_wrapper, model_type, vq_model,
                text_model=text_model, t2m_trans=t2m_trans, res_trans=res_trans,
                out_dir=args.out_dir, 
                repeat_time_mm=args.repeat_time_mm, time_steps=args.time_steps,
                t2m_tem=args.t2m_tem, t2m_cond_scale=args.t2m_cond_scale, 
                res_tem=args.res_tem, res_cond_scale=args.res_cond_scale)
        logger.info(f"Repeat time {i}:")
        for name in metrics.keys():
            logger.info(f"\t{name}: {_metrics[name]}")
            metrics[name].append(_metrics[name])

    logger.info("Final average:")
    for name in metrics.keys():
        metrics[name] = np.array(metrics[name])
        logger.info(f"\t{name}: {np.mean(metrics[name]):.3f}, conf. {np.std(metrics[name]) * 1.96 / np.sqrt(args.repeat_time):.3f}")

    #################################################################