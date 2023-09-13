import argparse
import time
import torch
from datetime import datetime
import os


def get_args(args, namespace=None):
    parser = initialise_arg_parser(args, 'Ssam.')

    # Experimental setup
    parser.add_argument("--task", type=str, help="Task to solve")

    # Utility
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--identifier", type=str, default=str(time.time()), help="Identifier for the current job")

    # Experiment configuration
    parser.add_argument("-ep", "--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--optimizer", default='sgd', type=str, help="Optimizer")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--test-batch-size", default=128, type=int, help="Test batch size")

    # SAM + Decay + Pruning
    parser.add_argument("--rho", default=0.1, type=float, help="SAM perturbation size")
    parser.add_argument("--sam-ord", default=2, choices=(1, 2, 'inf'), help="Order of perturbation size (l1, l2, or l-inf)")
    parser.add_argument("--scaled-max", default="gradnorm", choices=("gradnorm", "none", "adam"), help="Scaling of max step")
    parser.add_argument("-p", "--prune-threshold", default=2**-8, type=float, help="Pruning threshold")
    parser.add_argument("--decay-type", default='l1', type=str, help="Decay type (l1 or l2)")
    parser.add_argument("--decay-rate", default=0.1, type=float, help="Decay rate")

    # SETUP ARGUMENTS
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="../outputs/",
        help="Base root directory for the output."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/",
        help="Base root directory for the dataset."
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Define on which GPU (CPU or m1 chip for Macs) to run the model. If -1, use CPU."
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Num workers for dataset loading"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="How often to do validation."
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        default="INFO"
    )
    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")
    os.makedirs("../logs/", exist_ok=True)
    parser.add_argument(
        "--logfile",
        type=str,
        default=f"../logs/log_{now}.txt"
    )
    # ----------------------------------->
    os.makedirs("../logs/", exist_ok=True)

    args = parser.parse_args(namespace=namespace)
    transform_gpu_args(args)
    args.task = args.task.lower()
    args.optimizer = args.optimizer.lower()
    return args


def transform_gpu_args(args):
    """
    Transforms the gpu arguments to the expected format.
    """
    if args.gpu == "m1":
        args.device = "mps" if \
            torch.backends.mps.is_available() else 'cpu'
    elif args.gpu == "-1":
        args.device = "cpu"
    else:
        gpu_str_arg = args.gpu.split(',')
        if len(gpu_str_arg) > 1:
            raise NotImplementedError("Multiple GPUs not supported.")
        else:
            args.device = f"cuda:{args.gpu}" if \
                torch.cuda.is_available() else 'cpu'


def initialise_arg_parser(args, description):
    parser = argparse.ArgumentParser(args, description=description)
    return parser
