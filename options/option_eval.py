import argparse
import torch


def get_args_parser():
    parser = argparse.ArgumentParser(description='Evaluation',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # experiment
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id')
    
    parser.add_argument('--vq-name', type=str, default='', help='')
    parser.add_argument('--model-name-vq', type=str, default='best_fid', help='')
    
    parser.add_argument('--mtrans-name', type=str, default='', help='')
    parser.add_argument('--model-name-mtrans', type=str, default='best_fid', help='')
    
    parser.add_argument('--rtrans-name', type=str, default='', help='')
    parser.add_argument('--model-name-rtrans', type=str, default='best_fid', help='')
    
    parser.add_argument('--repeat-time', type=int, default=20, help='repeat times form evaluation')
    # parser.add_argument('--repeat-time-mm', type=int, default=30, help='repeat times form diversity')
    parser.add_argument('--repeat-time-mm', type=int, default=1, help='repeat times form diversity')

    parser.add_argument('--time-steps', type=int, default=10, help='')

    parser.add_argument('--t2m-tem', type=float, default=1, help='')
    parser.add_argument('--t2m-cond-scale', type=float, default=4, help='')
    parser.add_argument('--res-tem', type=float, default=1, help='')
    parser.add_argument('--res-cond-scale', type=float, default=5, help='')
    
    # dataloader
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')

    parser.add_argument('--clip-version', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument("--clip-dim", type=int, default=512, help="")

    args, unknown = parser.parse_known_args()
    if unknown:
        print("Warning: unknown arguments:", unknown)
        # exit(-1)

    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))
    print(f"Using Device: {args.device}")
    
    args.nb_joints = 21 if args.dataname == 'kit' else 22
    
    if args.dataname == 'kit':
        args.t2m_cond_scale = 2

    torch.autograd.set_detect_anomaly(True)

    return args
