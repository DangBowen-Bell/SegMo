import argparse
import torch


def get_args_parser():
    parser = argparse.ArgumentParser(description='RVQ-VAE',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # experiment
    parser.add_argument('--exp-name', type=str, default="exp_name", help='Name of the experiment')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu-id", type=int, default=0, help='GPU id')
    parser.add_argument('--out-dir', type=str, default='output/vq', help='results are saved here')
    parser.add_argument('--print-iter', default=50, type=int, help='iter log frequency')
    parser.add_argument('--save-iter', default=500, type=int, help='iter save latest model frequency')
    
    # dataloader
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset name') 
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')

    # training
    parser.add_argument('--max-epoch', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--warm-up-iter', default=2000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--vel', type=float, default=0.5, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l1_smooth', help='reconstruction loss')

    # model
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="num of resblocks for each res")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument('--vq-act', type=str, default='relu', choices=['relu', 'silu', 'gelu'], help='activation function')
    parser.add_argument('--vq-norm', type=str, default=None, help='normalization method')
    parser.add_argument('--num-quantizers', type=int, default=6, help='num_quantizers')
    parser.add_argument('--shared-codebook', action="store_true")
    parser.add_argument('--quantize-dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')

    args = parser.parse_known_args()[0]
    
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

    return args