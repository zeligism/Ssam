
import numpy as np
from copy import deepcopy
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from models.lenet import LeNet5
from models.resnet import ResNet9
from models.vit import SimpleViT
from optim_wrappers.decoupled_decay import DecoupledDecay
from optim_wrappers.masked_optim import MaskedOptimizer
from optim_wrappers.sam import SAM, SADAM
from utils import *

DATASET_DIR = "data"
TEST_AT_EPOCH_END = True
USE_CIFAR = False


def parse_args():
    ...


@torch.no_grad()
def test(model, loss_fn, test_loader, device, show_results=True, get_gradnorm=False):
    model.eval()

    def closure():
        test_loss = 0
        correct = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y) / len(test_loader)
            if get_gradnorm:
                loss.backward()
            test_loss += loss.item()
            # Accuracy
            pred = y_pred.max(dim=1).indices
            correct += (pred == y).sum().item()
        return test_loss, correct

    if get_gradnorm:
        closure = torch.enable_grad()(closure)
        test_loss, correct = closure()
        flat_grad = []
        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                flat_grad.append(p.grad.view(-1))
                p.grad = None
        gradnorm = torch.cat(flat_grad).pow(2).sum().sqrt().item()
    else:
        test_loss, correct = closure()
        gradnorm = 0.0

    sparsity = check_sparsity(model, show_results=False)
    acc = correct / len(test_loader.dataset)
    error = 1 - acc
    if show_results:
        print(f"\nTest set: Average loss: {test_loss:.4f}, "
              f"Accuracy: {correct}/{len(test_loader.dataset)} ({100. * acc:.0f}%), "
              f"Sparsity: {100. * sparsity:.2f}%\n")

    return test_loss, gradnorm, error, sparsity


def train(model, optimizer, loss_fn, train_loader, test_loader, device, epoch,
          log_interval=50, test_interval=0.25, get_gradnorm=False):
    model.train()
    data = []
    # Add test at initial point
    if epoch == 1:
        result0 = test(model, loss_fn, test_loader, device, show_results=True, get_gradnorm=get_gradnorm)
        data.append((0.,) + result0)

    # Training loop
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        def closure():
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        # Logging
        if batch_idx % log_interval == 0:
            batch_ratio = batch_idx / len(train_loader)
            sparsity = check_sparsity(model, show_results=False)
            print(f"Train Epoch: {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)} "
                  f"({100. * batch_ratio:.0f}%)]\tLoss: {loss.item():.6f},\tSparsity: {100. * sparsity:.2f}%")
        # Testing
        should_test = (batch_idx + 1) % round(test_interval * len(train_loader)) == 0
        last_epoch = batch_idx == len(train_loader) - 1
        if should_test or last_epoch:
            ep = epoch - 1 + (batch_idx + 1) / len(train_loader)
            result = test(model, loss_fn, test_loader, device, show_results=True, get_gradnorm=get_gradnorm)
            data.append((ep,) + result)

    return data


def init_device(cuda):
    use_cuda = torch.cuda.is_available() and cuda
    use_mps = torch.backends.mps.is_available() and cuda
    device = torch.device("cuda" if use_cuda else ("mps" if use_mps else "cpu"))
    if use_cuda:
        print(f"Using CUDA")
    if use_mps:
        print(f"Using MPS")


def init_seed(seed):
    if seed is not None:
        print(f"Setting random seed to {seed}.")
        np.random.seed(seed)
        torch.manual_seed(seed)


def init_model(experiment="lenet"):
    experiment = experiment.lower()
    if experiment == "lenet":
        model = LeNet5()

    elif experiment == "resnet":
        model = ResNet9()

    elif experiment == "vit":
        model = SimpleViT(image_size=32,
                    patch_size=16,  # patch_size=8,
                    num_classes=10,  # for cifar 10 and mnist
                    dim=128,
                    depth=12,
                    heads=8,
                    mlp_dim=64)

    else:
        raise NotImplementedError(f"experiment '{experiment}' not implemented.")

    return model


def load_dataset(experiment="lenet"):
    experiment = experiment.lower()
    if experiment == "lenet":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(DATASET_DIR, train=False, transform=transform)

    elif experiment in ("resnet", "vit"):
        if not USE_CIFAR:
            ### MNIST ###
            image_size = 32
            transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            train_dataset = datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(DATASET_DIR, train=False, transform=transform)
        else:
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
            train_dataset = datasets.CIFAR10(DATASET_DIR, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(DATASET_DIR, train=False, transform=test_transform)
    else:
        raise NotImplementedError(f"experiment '{experiment}' not implemented.")


    return train_dataset, test_dataset


def make_masked_and_decayed(Optimizer):
    class MaskedDecayedOptimizer(MaskedOptimizer, DecoupledDecay, Optimizer):
        def __repr__(self):
            return Optimizer.__name__ + "." + super().__repr__()
        pass
    return MaskedDecayedOptimizer


def init_experiment(args, Optimizer=torch.optim.SGD, **optim_args):
    device = init_device(args.cuda)
    
    # All runs start are init based on the model seed
    print(f"Initializing model with random seed {args.model_seed}.")
    init_seed(args.model_seed)
    model = init_model(args.experiment).to(device)

    # Load dataset and initialize dataloader
    init_seed(args.seed)
    train_dataset, test_dataset = load_dataset(args.experiment)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    MaskedDecayedOptimizer = make_masked_and_decayed(Optimizer)
    optimizer = MaskedDecayedOptimizer(model.parameters(), **optim_args)

    print(f"Model has {count_params(model)} parameters")

    return model, optimizer, loss_fn, train_loader, test_loader, device


def main(args):
    # TODO
    class Args:
        experiment = "lenet"  # choices = ("lenet", "resnet", "vit")
        cuda = True
        seed = 0
        model_seed = 0
        num_workers = 0
        lr = 3e-3
        batch_size = 32
        epochs = 7

    args = Args()

    args.experiment = "resnet"
    args.lr = 1e-3

    rho = 0.1  # relative to lr
    decay_rate = 0.1  # relative to lr
    prune_threshold = 2**-8
    decay_type = "l1"

    optim_args = dict(lr=args.lr, rho=rho, scaled_max="adam", sam_ord=2, #amsgrad=True,
                    decay_type=decay_type, decay_rate=decay_rate, prune_threshold=prune_threshold)

    model, optimizer, loss_fn, train_loader, test_loader, device = init_experiment(args, Optimizer=SADAM, **optim_args)

    data = []
    model_state_dict0 = deepcopy(model.state_dict())
    optimizer_state_dict0 = deepcopy(optimizer.state_dict())


if __name__ == "__main__":
    args = parse_args()
    main(args)
