
import torch


@ torch.no_grad()
def count_params(model):
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def get_flat_params(optimizer):
    return torch.cat([p.view(-1) for group in optimizer.param_groups for p in group['params']])


@torch.no_grad()
def get_flat_grads(optimizer):
    return torch.cat([p.grad.view(-1) for group in optimizer.param_groups for p in group['params'] if p.grad is not None])


@torch.no_grad()
def get_flat_exp_avg_sq(optimizer):
    return torch.cat([optimizer.state[p]["exp_avg_sq"].view(-1) for group in optimizer.param_groups for p in group['params'] if "exp_avg_sq" in optimizer.state[p]])


@torch.no_grad()
def set_flat_params(optimizer, flat_params):
    j = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            p.data.copy_(flat_params[j:j + p.numel()].reshape_as(p).data)
            j += p.numel()


@torch.no_grad()
def set_flat_grads(optimizer, flat_grads):
    j = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.data.copy_(flat_grads[j:j + p.numel()].reshape_as(p).data)
                j += p.numel()


@torch.no_grad()
def set_model_flat_params(model, flat_params):
    j = 0
    for p in model.parameters():
        p.data.copy_(flat_params[j:j + p.numel()].reshape_as(p).data)
        j += p.numel()


@torch.no_grad()
def check_sparsity(model, eps=1e-8, show_results=True):
    nonzeros = sum((p.abs() > eps).sum().item() for p in model.parameters())
    total_params = sum((p.numel() for p in model.parameters()))
    sparsity = 1 - nonzeros / total_params
    if show_results:
        print(f"Sparsity = {100 * sparsity:.2f}%")
    return sparsity



