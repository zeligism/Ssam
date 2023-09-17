
import os
import time
from argparse import Namespace
from itertools import product
from random import random, shuffle

from run import main
from opts import get_args
from utils.logger import Logger
from utils.utils import create_model_dir, save

DRY_RUN = False

HP_DICT = {
    "identifier": ["experiment1"],
    "task": ["lenet_mnist", "resnet_mnist", "resnet_cifar10", "vit_cifar10"],
    "gpu": ["0"],
    "num_workers": [2],
    # "seed": [0, 1, 2],
    "seed": [0],
    "epochs": [5],
    "batch_size": [32],
    # "lr": [1e-2, 3e-3, 1e-3],
    "lr": [1e-3],
    "optimizer": ["sgd", "sgd-mom", "adam"],
    "scaled_max": ["gradnorm", "none", "adam"],
    "rho": [0.01, 0.1, 1.0],
    "prune_threshold": [0.005, 0.05],
    "decay_type": ["l1"],
    "decay_rate": [0.1, 0.01],
}

HP_GRID = product(*HP_DICT.values())
# Comment out if you want experiments running in sequential order
# HP_GRID = list(HP_GRID)
# shuffle(HP_GRID)


def run_experiments():
    # Give other jobs a chance to avoid conflicts
    time.sleep(3 * random())

    for hp in HP_GRID:
        hp_dict = dict(zip(HP_DICT.keys(), hp))
        # Create arg namespace to pass to train
        args = get_args(None, namespace=Namespace(**hp_dict))

        model_dir = create_model_dir(args)
        data_path = os.path.join(model_dir, "data.pl")

        if os.path.exists(data_path):
            continue  # skip if another job already started on this

        try:
            open(data_path, 'a').close()
            main(args) if not DRY_RUN else print(args)
        except Exception as e:
            Logger.get().error(f"Encountered an error. Removing empty data file '{data_path}'.")
            os.remove(data_path)
            raise


if __name__ == "__main__":
    Logger.setup_logging("INFO")
    Logger()
    run_experiments()
