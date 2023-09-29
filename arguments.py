from os.path import join
import argparse
import time

import cnn.models as models
from cnn.utils.log import log
from cnn.utils.auxiliary import info2path

def get_args():
    parser = argparse.ArgumentParser(
        description='Training DNNs with HBFP')
    parser.add_argument('--type', type=str, default='getting_started', choices=['getting_started', 'cnn', 'lstm', 'bert'])
    # parse args.
    args, unknown = parser.parse_known_args()
    return args


def tutorial_args():
    parser = argparse.ArgumentParser(
        description='Getting Started')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--num_format', default='fp32', type=str,
                        help='number format for fully connected and convolutional layers')
    parser.add_argument('--rounding_mode', default='stoc', type=str,
                        help='Rounding mode for bfp')
    parser.add_argument('--mant_bits', default=8, type=int,
                        help='Mantissa bits for bfp')
    parser.add_argument('--bfp_tile_size', default=0, type=int,
                        help='Tile size if using tiled bfp. 0 disables tiling')
    parser.add_argument('--weight_mant_bits', default=0, type=int,
                        help='Mantissa bits for weights bfp')
    parser.add_argument('--posit_length', default=8, type=int,
                        help='Posit length')
    parser.add_argument('--es', default=2, type=int,
                        help='Exponent Size for Posit')
    parser.add_argument('--fixed_regime_length', default=4, type=int,
                        help='Regime length for Posit')

    # parse args.
    args, unknown = parser.parse_known_args()
    if args.weight_mant_bits == 0:
        args.weight_mant_bits = args.mant_bits
    return args

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_args(args):
    print('parameters: ')
    for arg in vars(args):
        print(arg, getattr(args, arg))

def log_args(args):
    log('parameters: ')
    for arg in vars(args):
        log(str(arg) + '\t' + str(getattr(args, arg)))

if __name__ == '__main__':
    args = get_args()
