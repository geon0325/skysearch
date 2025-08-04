import os
import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='comsX2')
    parser.add_argument('--num_workers', default=8, type=int)
    
    parser.add_argument('--n_train', default=12, type=int)
    parser.add_argument('--n_test', default=12, type=int)

    # model parameters
    # [10, 1, 64, 64] for mmnist
    # [4, 2, 32, 32] for taxibj  
    # [12, 1, 300, 375] for coms
    parser.add_argument('--in_shape', default=[12, 1, 600, 748], type=int,nargs='*') 
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=256, type=int)
    parser.add_argument('--N_S', default=2, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--save_step', default=1000, type=int)
    parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
    parser.add_argument('--ld', default=10.0, type=float, help='Lambda for weigting MSELoss compared to GANLoss')
    
    parser.add_argument('--init_epoch', default=0, type=int)
    parser.add_argument('--train_step', default=0, type=int)
    parser.add_argument('--aug_n', default=2.0, type=float)
    
    parser.add_argument('--mse_step', default=60, type=int)
    parser.add_argument('--gen_step', default=10, type=int)
    parser.add_argument('--disc_step', default=30, type=int)
    
    #parser.add_argument('--corrupt_ratio', default=0.1, type=float)
    
    parser.add_argument('--lmbda_feat', default=1.0, type=float)
    parser.add_argument('--lmbda_orth', default=1.0, type=float)
    
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
   
