
import sys
import os
import numpy as np
import random

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from opts import get_args
from train import train
from utils.utils import count_params, create_model_dir, save
from utils.logger import Logger

from models.lenet import LeNet5
from models.resnet9 import ResNet9
from models.vit import SimpleViT
from optim_wrappers.decoupled_decay import DecoupledDecay
from optim_wrappers.masked_optim import MaskedOptimizer
from optim_wrappers.sam import SAM, SADAM

OPTIMIZER_LOOKUP = {
    "sgd": SAM,
    "sgd-mom": SAM,
    "sam": SAM,
    "adam": SADAM,
    "sadam": SADAM,
}


def init_model(task="lenet"):
    if task.startswith("lenet"):
        model = LeNet5()

    elif task.startswith("resnet"):
        model = ResNet9()

    elif task.startswith("vit"):
        # XXX
        model = SimpleViT(image_size=32,
                          patch_size=16,  # patch_size=8,
                          num_classes=10,  # for cifar 10 and mnist
                          dim=128,
                          depth=12,
                          heads=8,
                          mlp_dim=64)
    else:
        raise NotImplementedError(f"Task '{task}' not implemented.")

    return model


def load_dataset(root, task="lenet"):
    if task.startswith("lenet"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root, train=False, transform=transform)

    elif task.startswith("resnet") or task.startswith("vit"):
        if 'mnist' in task:
            ### MNIST ###
            image_size = 32
            transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root, train=False, transform=transform)
        elif 'cifar' in task:
            ### CIFAR-10 ###
            cifar_mean = (0.4914, 0.4822, 0.4465)
            cifar_std = (0.2023, 0.1994, 0.2010)
            train_transform = transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar_mean, cifar_std)
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar_mean, cifar_std)
            ])
            train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(root, train=False, transform=test_transform)
    else:
        raise NotImplementedError(f"Task '{task}' not implemented.")

    return train_dataset, test_dataset


def make_masked_and_decayed(Optimizer):
    class MaskedDecayedOptimizer(MaskedOptimizer, DecoupledDecay, Optimizer):
        def __repr__(self):
            return Optimizer.__name__ + "." + super().__repr__()
        pass
    return MaskedDecayedOptimizer


def init_experiment(args):
    device = torch.device(args.device)
    Logger.get().info(f"Device = {device}")

    if torch.cuda.device_count():
        cuda_support = True
    else:
        Logger.get().warning('CUDA unsupported!!')
        cuda_support = False

    if args.seed is not None:
        if cuda_support:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)

    # ---------- Model ---------- #
    model = init_model(args.task).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    Logger.get().info(f"Model has {count_params(model)} parameters")

    # ---------- Data ---------- #
    train_dataset, test_dataset = load_dataset(args.data_path, args.task)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers)

    # ---------- Optimizer ---------- #
    optim_args = dict(lr=args.lr, rho=args.rho,
                      scaled_max=args.scaled_max, sam_ord=args.sam_ord,
                      decay_type=args.decay_type, decay_rate=args.decay_rate,
                      prune_threshold=args.prune_threshold)
    if args.optimizer.startswith("sgd") and "mom" in args.optimizer:
        optim_args["momentum"] = 0.9
    if args.optimizer not in OPTIMIZER_LOOKUP:
        raise NotImplementedError(args.optimizer)
    Optimizer = OPTIMIZER_LOOKUP[args.optimizer]
    MaskedDecayedOptimizer = make_masked_and_decayed(Optimizer)
    optimizer = MaskedDecayedOptimizer(model.parameters(), **optim_args)

    return model, optimizer, loss_fn, train_loader, test_loader, device


def main(args):
    model_dir = create_model_dir(args)
    model, optimizer, loss_fn, train_loader, test_loader, device = init_experiment(args)

    data = []  # Data = [(epoch, loss, gradnorm, accuracy, sparsity)]
    for epoch in range(1, args.epochs + 1):
        results = train(model, optimizer, loss_fn, train_loader, test_loader, device, epoch)
        data += results

    save(data, os.path.join(model_dir, "data.pl"))

if __name__ == "__main__":
    args = get_args(sys.argv)
    Logger.setup_logging(args.loglevel, logfile=args.logfile)
    Logger()
    main(args)
