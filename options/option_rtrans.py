import argparse
import torch


def get_args_parser():
    parser = argparse.ArgumentParser(description='Residual Transformer',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # experiment
    parser.add_argument('--exp-name', type=str, default='exp_name', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--out-dir', type=str, default='output/rtrans', help='output directory')    
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--vq-name', type=str, default='', help='')
    parser.add_argument('--model-name-vq', type=str, default='best_fid', help='')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id')
    parser.add_argument('--print-iter', default=50, type=int, help='print frequency')
    parser.add_argument('--save-iter', default=500, type=int, help='iter save latest model frequency')    
    parser.add_argument('--eval-epoch', default=10, type=int, help='evaluation frequency')
    
    parser.add_argument('--resume-ckpt', type=str, default='')

    # dataloader
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
  
    # training
    parser.add_argument('--max-epoch', type=int, default=500, help='maximum number of epoch for training')
    parser.add_argument('--warm-up-iter', default=2000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--milestones', default=[50_000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--lr-scheduler-type', type=str, default='multi_step')
    parser.add_argument('--lr-min', default=1e-6, type=float, help='')
    
    # useless
    parser.add_argument('--lr-scheduler', default=[150000], nargs="+", type=int, help="learning rate schedule (iterations)")

    # t2m
    parser.add_argument('--latent-dim', type=int, default=384, help='dimension of transformer latent.')
    parser.add_argument('--nb-head', type=int, default=6, help='number of heads.')
    parser.add_argument('--nb-layer', type=int, default=8, help='number of attention layers.')
    parser.add_argument('--ff-size', type=int, default=1024, help='FF_Size')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio in transformer')
    parser.add_argument('--cond-drop-prob', type=float, default=0.2, help='drop ratio of condition, for classifier-free guidance')
    parser.add_argument('--share-weight', action="store_false", help='whether to share weight for projection/embedding, for residual transformer.')

    # text model
    parser.add_argument('--clip-version', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument("--clip-dim", type=int, default=512, help="")

    # evaluation
    parser.add_argument('--time-steps', type=int, default=10, help='')
    parser.add_argument('--t2m-tem', type=float, default=1, help='')
    parser.add_argument('--t2m-cond-scale', type=float, default=4, help='')
    parser.add_argument('--res-tem', type=float, default=1, help='')
    parser.add_argument('--res-cond-scale', type=float, default=5, help='')

    parser.add_argument('--mtrans-name', type=str, default='', help='')
    parser.add_argument('--model-name-mtrans', type=str, default='best_fid', help='')
    
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Warning: unknown arguments:", unknown)
        # exit(-1)
   
    if args.gpu_id == -1 or not torch.cuda.is_available():
        args.device = torch.device("cpu")
    else:
        num_gpus = torch.cuda.device_count()
        if args.gpu_id >= num_gpus:
            print(f"Device {args.gpu_id} does not exist, using cuda:0")
            args.gpu_id = 0
        args.device = torch.device(f"cuda:{args.gpu_id}")
    print(f"Using Device: {args.device}")
    
    args.nb_joints = 21 if args.dataname == 'kit' else 22

    if args.dataname == 'kit':
        args.t2m_cond_scale = 2

    torch.autograd.set_detect_anomaly(True)
    
    return args