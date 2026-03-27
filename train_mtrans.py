import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.data import DataLoader

from options.option_mtrans import get_args_parser
from dataset.dataset_t2m import Text2MotionDataset
from models.trans import MaskTransformer
from models.trainer import MaskTransformerTrainer
from utils import new_utils


if __name__ == '__main__':
    args = get_args_parser()
    args.is_train = True
    
    new_utils.fixseed(args.seed)
    
    if args.resume_ckpt == '':
        new_utils.init_output_dir(args)
        new_utils.save_config(args)
    else:
        args.out_dir = f"{args.out_dir}_{args.dataname}/{args.resume_ckpt}"

    eval_wrapper = new_utils.load_eval_wrapper(args.dataname)
    
    if args.vq_name == '':
        vq_model, args_vq = new_utils.load_rvqvae_old(args)
    else:
        vq_model, args_vq = new_utils.load_rvqvae(args)
    args.code_dim = args_vq.code_dim
    args.nb_code = args_vq.nb_code
    args.down_t = args_vq.down_t

    t2m_trans = MaskTransformer(
        code_dim=args.code_dim,
        cond_mode='text',
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.nb_layer,
        num_heads=args.nb_head,
        dropout=args.dropout,
        clip_dim=args.clip_dim,
        cond_drop_prob=args.cond_drop_prob,
        args=args)
    
    text_model = new_utils.load_text_model(args)

    collate_fn = new_utils.collate_fn_t2m
    collate_fn_eval = new_utils.collate_fn_t2m_eval
    
    train_set = Text2MotionDataset(
        args.dataname, 
        False,
        unit_length=2**args.down_t)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=1, 
        shuffle=True, 
        drop_last=True,
        collate_fn=collate_fn)
    
    val_set = Text2MotionDataset(
        args.dataname, 
        False,
        usage='val',
        unit_length=2**args.down_t)
    
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        num_workers=1, 
        shuffle=True, 
        drop_last=True,
        collate_fn=collate_fn)
    
    eval_val_set = Text2MotionDataset(
        args.dataname, 
        True,
        usage='val',
        unit_length=2**args.down_t)
    
    eval_val_loader = DataLoader(
        eval_val_set, 
        batch_size=32, 
        num_workers=4,
        shuffle=True, 
        drop_last=True,
        collate_fn=collate_fn_eval)

    trainer = MaskTransformerTrainer(args, t2m_trans, vq_model, text_model)
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper)