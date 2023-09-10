
import torch
import numpy as np
import matplotlib.pyplot as plt


def smoothed(arr):
    arr = np.array(arr)
    cumsum_vec = np.cumsum(np.insert(arr, 0, 0))
    window_width = len(cumsum_vec) // 100
    smoothed_arr = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return smoothed_arr


@torch.no_grad()
def plot_stats(optimizer, data=None, num_dims=2):
    # p1 = torch.cat([p.view(-1) for group in optimizer.param_groups for p in group['params']]).cpu().abs()
    # g1 = torch.cat([optimizer.state[p]['exp_avg'].view(-1) for group in optimizer.param_groups for p in group['params']]).cpu().abs()
    # g2 = torch.cat([optimizer.state[p]['exp_avg_sq'].view(-1) for group in optimizer.param_groups for p in group['params']]).cpu().abs()

    if data is not None:
        plt.figure()
        plt.semilogy(np.array(data)[:,1])
        plt.title("loss")
        plt.figure()
        plt.plot(np.array(data)[:,4])
        plt.title("sparsity")

    plt.figure()
    plt.plot(smoothed(optimizer.sharpness[10:]))
    plt.plot([0, len(optimizer.sharpness)], [0.0, 0.0], color='r', linestyle='dashed')
    plt.title("sharpness")
    
    plt.figure()
    for group in optimizer.param_groups:
        for p in group['params']:
            if len(p.shape) == num_dims:
                p1 = p.cpu()
                plt.hist(p1[p1.abs() > 1e-3], bins=50, alpha=0.5)
    plt.title("params")

    plt.figure()
    for group in optimizer.param_groups:
        for p in group['params']:
            if len(p.shape) == num_dims:
                g1 = optimizer.state[p]["exp_avg"].cpu()
                plt.hist(g1[g1.abs() > 1e-4], bins=50, alpha=0.5)
    plt.title("exp_avg")

    plt.figure()
    for group in optimizer.param_groups:
        for p in group['params']:
            if len(p.shape) == num_dims:
                g2 = optimizer.state[p]["exp_avg_sq"].sqrt().cpu().abs()
                plt.hist(g2[g2 > 0].log10(), bins=50, alpha=0.5)
    plt.title("log(exp_avg_sq)")

    plt.figure()
    for group in optimizer.param_groups:
        for p in group['params']:
            if len(p.shape) == num_dims:
                pg = (p * optimizer.state[p]["exp_avg"]).cpu()
                plt.hist(pg[pg != 0], bins=50, alpha=0.5)
    plt.title("p * m")
    plt.figure()
    for group in optimizer.param_groups:
        for p in group['params']:
            if len(p.shape) == num_dims:
                pg = (p * optimizer.state[p]["exp_avg"] / optimizer.state[p]["exp_avg_sq"].sqrt().add(1e-8)).cpu()
                plt.hist(pg[pg != 0], bins=50, alpha=0.5)
    plt.title("p * m/sqrt(v)")


