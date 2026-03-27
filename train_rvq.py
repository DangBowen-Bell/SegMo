import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader

from options.option_rvq import get_args_parser
from dataset.dataset_rvq import MotionDataset
from dataset.dataset_t2m import Text2MotionDataset
from models.rvqvae import RVQVAE
from models.trainer import RVQVAETrainer
from utils import new_utils


if __name__ == "__main__":
    args = get_args_parser()
    args.is_train = True
    
    new_utils.fixseed(args.seed)
    
    new_utils.init_output_dir(args)
    new_utils.save_config(args)

    eval_wrapper = new_utils.load_eval_wrapper(args.dataname)
    
    vq_model = RVQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm)

    train_set = MotionDataset(
        args.dataname, 
        window_size=args.window_size,
        unit_length=2**args.down_t)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size, 
        num_workers=1,
        shuffle=True, 
        drop_last=True,
        pin_memory=True)

    val_set = MotionDataset(
        args.dataname,
        usage='val',
        window_size=args.window_size,
        unit_length=2**args.down_t)

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size, 
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True)

    collate_fn_eval = new_utils.collate_fn_t2m_eval

    eval_val_set = Text2MotionDataset(
        args.dataname, 
        True,
        usage='val',
        unit_length=2**args.down_t)

    eval_val_loader = DataLoader(
        eval_val_set, 
        batch_size=32, 
        num_workers=1, 
        shuffle=True, 
        drop_last=True,
        collate_fn=collate_fn_eval)
    
    trainer = RVQVAETrainer(args, vq_model=vq_model)
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper)