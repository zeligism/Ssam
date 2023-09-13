
import torch
from utils.utils import check_sparsity
from utils.logger import Logger


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
        Logger.get().info(
            f"Test set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(test_loader.dataset)} ({100. * acc:.0f}%), "
            f"Sparsity: {100. * sparsity:.2f}%")

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
            Logger.get().info(
                f"Train Epoch: {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)} "
                f"({100. * batch_ratio:.0f}%)] Loss: {loss.item():.6f},\tSparsity: {100. * sparsity:.2f}%")
        # Testing
        should_test = (batch_idx + 1) % round(test_interval * len(train_loader)) == 0
        last_epoch = batch_idx == len(train_loader) - 1
        if should_test or last_epoch:
            ep = epoch - 1 + (batch_idx + 1) / len(train_loader)
            result = test(model, loss_fn, test_loader, device, show_results=True, get_gradnorm=get_gradnorm)
            data.append((ep,) + result)

    return data
