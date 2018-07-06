import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='MNIST | Fashion-MNIST | CIFAR10 | CelebA')
parser.add_argument('--dataroot', default = '/data/jehyuk/imgdata/', help='absolute path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda or not')
parser.add_argument('--cuda_index', default='0', help='number of cuda usages')
parser.add_argument('--random_seed', default=100, help='random_seed')

parser.add_argument('--model', type=str, default='BEGAN', help = 'A kind of model you want to do')
parser.add_argument('--is_train', type=bool, default=True, help='Training mode vs inference mode')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('--batchsize', type=int, default=16, help='batchsize')
parser.add_argument('--maxepoch', type=int, default=20, help='n_epochs')
parser.add_argument('--imagesize', type=int, default=64)
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate of G')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate of D')
parser.add_argument('--channel_bunch', type=int, default=64, help='increasing convolution channel bunch')
parser.add_argument('--n_z', type=int, default=64, help = 'latent vector z size')
parser.add_argument('--gamma', type=float, default=0.75, help='Convergence measure')

parser.add_argument('--result_dir', default = '/home/jehyuk/GenerativeModels/GAN/results/')
parser.add_argument('--save_dir', default = '/home/jehyuk/GenerativeModels/GAN/models/')
parser.add_argument('--n_sample', default = 36, type = int)

def get_config():
    config = parser.parse_args()
    return config