import os
from typing import (
    List,
    Dict
)

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_results(y, y_hat):
    fig, ax = plt.subplots(10, 3, figsize=(30, 25), facecolor="white")
    for i in range(10):
        for j in range(3):
            ax[i, j].plot(y_hat[:, 3 * i + j], color="red", label="predicted values", alpha=0.5)
            ax[i, j].plot(y[:, 3 * i + j], color="green", label="ground truth", alpha=0.75)
            ax[i, j].legend(loc=0)
            ax[i, j].title.set_text(f"Prediction error for link {3 * i + j}")
    plt.tight_layout()
    img = fig2img(fig)
    return img

def plot_results(y, y_hat, num_flow=None):
    if len(y.shape) > 1:
        num_flows = y.size(0)
        fig, axes = plt.subplots(num_flows, 1, figsize=(8, 2*num_flows))
        for i in range(num_flows):
            axes[i].plot(y[i].numpy(), color='green', label='Ground Truth')
            axes[i].plot(y_hat[i].numpy(), color='blue', label='Prediction')
            axes[i].set_title(f'Flow {i}')
            axes[i].legend()
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 2))
        axes.plot(y.numpy(), color='green', label='Ground Truth')
        axes.plot(y_hat.numpy(), color='blue', label='Prediction')
        axes.set_title(f'Flow {num_flow}')
        axes.legend()
    plt.tight_layout()
    img = fig2img(fig)
    return img


def plot_path_results(y, y_hat, status=None):
    fig, ax = plt.subplots(44, 3, figsize=(30, 120), facecolor="white")
    for i in range(44):
        for j in range(3):
            ax[i, j].plot(y_hat[:, 3 * i + j], color="blue", label="predicted values", alpha=0.5)
            ax[i, j].plot(y[:, 3 * i + j], color="green", label="ground truth", alpha=0.75)
            ax[i, j].legend(loc=0)
            if status is not None:
                ax2 = ax[i, j].twinx()
                status_str = list(map(lambda l: "anomalous" if l == 1 else "regular", status[3 * i + j, :]))
                ax2.plot(status_str, color="red")
                ax2.set_yticks([0, 1], ["regular", "anomalous"])
            ax[i, j].title.set_text(f"Prediction error for path {3 * i + j}")
    plt.tight_layout()
    img = fig2img(fig)
    return img


def plot_single_path_result(y, y_hat, missing_path, status=None):
    fig, ax = plt.subplots(1, 1, facecolor="white")
    ax.plot(y_hat, color="blue", label="predicted values", alpha=0.5)
    ax.plot(y, color="green", label="ground truth", alpha=0.75)
    ax.legend(loc=0)
    if status is not None:
        ax2 = ax.twinx()
        status_str = list(map(lambda l: "anomalous" if l == 1 else "regular", status))
        ax2.plot(status_str, color="red")
        ax2.set_yticks([0, 1], ["regular", "anomalous"])
    ax.title.set_text(f"Prediction error for path {missing_path}")
    plt.tight_layout()
    img = fig2img(fig)
    return img


def plot_histograms(link_errors: List[List[int]]): 
    fig, ax = plt.subplots(10, 3, figsize=(30, 25))
    for i in range(10):
        for j in range(3):
            ax[i, j].hist(link_errors[3 * i + j], edgecolor="black")
            ax[i, j].legend(loc=0)
            ax[i, j].title.set_text(f"Histogram of prediction errors for link {3 * i + j}")
    plt.tight_layout()
    img = fig2img(fig)
    return img


def anomaly_detections(y, y_hat, status, threshold=40):
    fig, ax = plt.subplots(44, 3, figsize=(30, 120), facecolor="white")
    for i in range(44):
        for j in range(3):
            ax[i, j].plot(y_hat[3 * i + j, :], color="blue", label="predicted values", alpha=0.4)
            ax[i, j].plot(y[3 * i + j, :], color="green", label="ground truth", alpha=0.4)
            ax[i, j].legend(loc=0)
            ax2 = ax[i, j].twinx()
            status_str = list(map(lambda l: "anomalous" if l == 1 else "regular", status[3 * i + j, :]))
            ax2.plot(status_str, color="red", label="status", alpha=.75)
            relative_error = ((y[3 * i + j, :] - y_hat[3 * i + j, :]).abs() / y[3 * i + j, :].abs()) * 100
            detections = torch.zeros(y[3 * i + j, :].shape)
            detections[relative_error > threshold] = 1
            detections_str = list(map(lambda l: "anomalous" if l == 1 else "regular", detections))
            ax2.plot(detections_str, color="blue", linestyle="dashed", label="detected status", alpha=.75)
            ax2.set_yticks([0, 1], ["regular", "anomalous"])
            ax[i, j].title.set_text(f"Detections for path {3 * i + j} ({threshold}% threshold)")
    plt.tight_layout()
    img = fig2img(fig)
    return img


def anomaly_single_path_detections(y, y_hat, status, missing_path, threshold=40):
    fig, ax = plt.subplots(1, 1)
    ax.plot(y_hat, color="blue", label="predicted values", alpha=0.4)
    ax.plot(y, color="green", label="ground truth", alpha=0.4)
    ax2 = ax.twinx()
    status_str = list(map(lambda l: "anomalous" if l == 1 else "regular", status))
    ax2.plot(status_str, color="red", label="status", alpha=.75)
    relative_error = ((y - y_hat).abs() / y.abs()) * 100
    detections = torch.zeros(y.shape)
    detections[relative_error > threshold] = 1
    detections_str = list(map(lambda l: "anomalous" if l == 1 else "regular", detections))
    ax2.plot(detections_str, color="blue", linestyle="dashed", label="detected status", alpha=.75)
    ax2.set_yticks([0, 1], ["regular", "anomalous"])
    ax2.legend()
    ax.title.set_text(f"Detections for path {missing_path} ({threshold}% threshold)")
    plt.tight_layout()
    img = fig2img(fig)
    return img


def plot_path_histograms(path_errors: List[int], path_index):
    fig, ax = plt.subplots(1, 1)
    ax.hist(path_errors, edgecolor="black")
    ax.legend(loc=0)
    ax.title.set_text(f"Path {path_index}")
    plt.tight_layout()
    img = fig2img(fig)
    return img


def plot_CDF(data, title, zoomed=False):
    np_data = np.array(data)
    count, bins_count = np.histogram(np_data, bins=100000, range=(np.nanmin(np_data), np.nanmax(np_data)))
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    fig, ax = plt.subplots(1, 1)
    if zoomed:  # retain 95% of CDF
        index = np.where(cdf > 0.95)[0][0]
        ax.plot(bins_count[:index], cdf[:index], label="CDF")
        ax.title.set_text(f"CDF of {title} (zoomed to 95% of CDF)")
    else:
        ax.plot(bins_count[:-1], cdf, label="CDF")
        ax.title.set_text(f"CDF of {title}")
    img = fig2img(fig)
    return img


def plot_pdfs(y, y_hat):
    values = [((y - y_hat) ** 2).mean(dim=1),  # MSE
              ((y - y_hat).abs() / y.abs()).mean(dim=1) * 100,  # MRE
              (((y - y_hat).abs() / y.abs()) * 100).reshape(-1)]  # RE
    titles = ["Mean Square Error (MSE)",
              "Mean Relative Error (MRE)",
              "Relative Errors (RE)"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), facecolor="white")
    for i in range(3):
        values[i][values[i] == np.inf] = np.nan  # replace inf with nans
        count, bins_count = np.histogram(values[i], bins=20, range=(np.nanmin(values[i]), np.nanmax(values[i])))
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        ax[i].plot(bins_count[1:], pdf, color="red", label="PDF")
        ax[i].plot(bins_count[1:], cdf, label="CDF")
        ax[i].legend()
        ax[i].title.set_text(titles[i])
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
    plt.suptitle("PDFs and CDFs of performance metrics")
    img = fig2img(fig)
    return img


def log_weights(tb, model, step):
    for name, value in model.named_parameters():
        try:
            tb.add_histogram(name, value.cpu(), step)
        except ValueError:
            print(f"Empty histogram error. Values: {value.cpu()}")


def log_grads(tb, model):
    for name, value in model.named_parameters():
        tb.add_histogram(f"{name}/grad", value.grad.cpu())


def save_predictions(y, y_hat, dataset="train"):
    if os.path.exists("predictions.pth"):
        data = torch.load("predictions.pth")
    else:
        data = dict(
            y_train=[],
            y_hat_train=[],
            y_test=[],
            y_hat_test=[]
        )
    data[f"y_{dataset}"].append(y)
    data[f"y_hat_{dataset}"].append(y_hat)
    torch.save(
        data,
        "predictions.pth"
    )


def save_alphas(alphas, file_name, dataset="train"):
    if os.path.exists(f"{file_name}.pth"):
        data = torch.load(f"{file_name}.pth")
    else:
        data = dict(
            alphas_train=[],
            alphas_test=[]
        )
    data[f"alphas_{dataset}"] = alphas
    torch.save(
        data,
        f"{file_name}.pth"
    )


def plot_alphas(alphas: List[torch.Tensor], path_indices: Dict, missing_path: int, edge_index: torch.Tensor,
                path_groups: Dict):
    path_indices_inv = {v: k for k, v in path_indices.items()}
    alphas = torch.stack(alphas).mean(dim=0)
    mask = edge_index[1] == missing_path
    incoming_paths = edge_index[0][mask]
    alphas_missing_path = alphas[mask]
    values, indices = alphas_missing_path.sort(descending=True)
    fig, ax = plt.subplots(1, 1)
    groups = [path_groups[path.item()] for path in indices]
    groups_str = [f"Group {group}" for group in groups[:40]]
    colors = ["red", "blue", "green", "yellow", "orange"]
    colors = [colors[group - 1] for group in groups[:40]]
    bars = ax.barh(
        [f"({path_indices_inv[incoming_paths[path.item()].item()][0]}, "
         f"{path_indices_inv[incoming_paths[path.item()].item()][1]})" for path in indices[:40]],
        [round(value.item(), 4) for value in values[:40]],
        color=colors,
        label=groups_str
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    # ax.bar_label(bars)
    plt.suptitle(f"Alphas for missing path {missing_path} (group {path_groups[missing_path]}, "
                 f"({path_indices_inv[missing_path][0]}, {path_indices_inv[missing_path][1]}) )")
    plt.tight_layout()
    img = fig2img(fig)
    return img


def alphas_histograms(alphas: List[torch.Tensor], missing_path: int, edge_index: torch.Tensor, path_groups: Dict):
    alphas = torch.stack(alphas).mean(dim=0)
    mask = edge_index[1] == missing_path
    incoming_paths = edge_index[0][mask]  # the paths corresponding to each alpha value
    alphas_missing_path = alphas[mask].cpu()  # the alpha values
    paths_per_group = {group: [] for group in set(path_groups.values())}
    for path in incoming_paths:
        paths_per_group[path_groups[path.item()]].append(path.item())
    fig, ax = plt.subplots(1, 1)
    colors = ["red", "blue", "green", "yellow", "orange"]
    for group in paths_per_group.keys():
        ax.hist(
            alphas_missing_path[torch.isin(incoming_paths, torch.tensor(paths_per_group[group]))],
            label=f"Group {group}",
            color=colors[group - 1],
            alpha=.5
        )
    ax.legend()
    plt.suptitle(f"Alphas histograms for missing path {missing_path} (group {path_groups[missing_path]})")
    plt.tight_layout()
    img = fig2img(fig)
    return img
