
import platform

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist
import cnn
import os

from arguments import tutorial_args
# from cnn.utils.log import log, configure_log
# from cnn.utils.set_conf import set_conf
# from cnn.models.create_model import create_model
# from cnn.runs.distributed_running import train_and_validate as train_val_op
# from lstm.train_and_val import train_lstm
from getting_started.resnet_cifar10 import resnet18_cifar10
from torch.multiprocessing import Process
import pdb
# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)


def main(args):
    
    if args.type == 'getting_started':
        args = tutorial_args()
        resnet18_cifar10(args)

def run(rank, size):
    """ Distributed Synchronous SGD Example """
    args = get_cnn_args()
    set_conf(args)
    print('set_conf...')
    # create model and deploy the model.
    model, criterion, optimizer = create_model(args)
    # config and report.
    configure_log(args)
    print('configure_log...')
    log_args(args)
    print('log_args...')

    device = 'GPU-'+ str(torch.cuda.current_device()) if args.device != "cpu" else "cpu"

    log(
        'Rank {} {}'.format(
            args.cur_rank,
            device
            # args.cur_gpu_device
            )
        )

    train_val_op(args, model, criterion, optimizer)


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    args = get_args()
    main(args)
