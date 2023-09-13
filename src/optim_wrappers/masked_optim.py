
import torch
from utils.utils import *

class MaskedOptimizer(torch.optim.Optimizer):
    def __init__(self, *args, prune_threshold=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = {}
        self.prune_threshold = prune_threshold
        self.use_masked_closure = False  # TODO

    def __repr__(self):
        return "Masked." + super().__repr__()

    def _parameters(self):
        for group in self.param_groups:
            for param in group['params']:
                yield param

    @torch.no_grad()
    def update_mask(self):
        for p in self._parameters():
            if p in self.mask:
                self.mask[p] &= p != 0.
            else:
                self.mask[p] = p != 0.

    @torch.no_grad()
    def mask_params(self):
        for p in self._parameters():
            if p in self.mask:
                p[~self.mask[p]] = 0.
                # XXX: mask out moments?
                if "exp_avg" in self.state[p]:
                    self.state[p]["exp_avg"][~self.mask[p]] = 0.
                if "exp_avg_sq" in self.state[p]:
                    self.state[p]["exp_avg_sq"][~self.mask[p]] = 0.

    @torch.no_grad()
    def mask_grads(self):
        for p in self._parameters():
            if p in self.mask[p] and p.grad is not None:
                p.grad.mul_(self.mask[p])

    @torch.no_grad()
    def prune_smallest(self, sparsity=0.9):
        flat_params = get_flat_params(self)
        sparse_elems = max(1, round((1 - sparsity) * flat_params.numel()))
        threshold = torch.topk(flat_params.abs().cpu(), k=sparse_elems).values[-1]
        flat_params[flat_params.abs() < threshold] = 0.
        set_flat_params(self, flat_params)
        self.update_mask()

    @torch.no_grad()
    def prune_smaller_than(self, threshold=None):
        for p in self._parameters():
            threshold = self.prune_threshold
            # XXX: prune absolutely or prop to sqrt(v)?
            # if "exp_avg_sq" in self.state[p]:
            #     threshold += self.state[p]["exp_avg_sq"].sqrt()
            p[p.abs() < threshold] = 0.
        self.update_mask()

    @torch.no_grad()
    def masked_closure(self, closure):
        closure = torch.enable_grad()(closure)

        def closure_then_mask_grads():
            closure()
            self.mask_grads()

        return closure_then_mask_grads

    def step(self, closure):
        if self.use_masked_closure:
            closure = self.masked_closure(closure)
        loss = super().step(closure)  # assuming first-order step
        self.prune_smaller_than()
        self.mask_params()
        return loss
