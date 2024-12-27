import os
import argparse
from dataclasses import dataclass

from utils import (
    log_weights,
    plot_results
)
from dataset import (
    read_data,
    read_wadi_data
)
from model import GAT_AD

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
import git
import wandb

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--name", default="", type=str, help="wandb logging name suffix")
parser.add_argument("--epochs", default=100, type=int, help="number of epochs to train")
parser.add_argument("--device", default=0, type=int, help="CUDA device to use")
parser.add_argument("--dropout", default=0, type=float, help="dropout to use in the model layers")
parser.add_argument("--outf", default="checkpoint", type=str, help="path to store the checkpoint")
parser.add_argument("--loss_func", default="MSE", type=str, help="loss function to use")
parser.add_argument("--learning_rate", default=0.001, type=float, help="initial learning rate")
parser.add_argument("--sched_step", default=50, type=int, help="steps for every learning rate reduction")
parser.add_argument("--wandb", default=False, action="store_true", help="toggle wandb logging")
parser.add_argument("--optim", default="Adam", type=str, help="optimizer to use")
parser.add_argument("--cpu", default=False, action="store_true", help="force use of CPU for training")
parser.add_argument("--log", default=False, action="store_true", help="apply log transform to data before inference")
parser.add_argument("--batch_size", default=32, type=int, help="set batch size")
parser.add_argument("--window_size", default=5, type=int, help="set window size")
parser.add_argument("--att_temp", default=1, type=float, help="temperature to use before passing alphas through softmax")
parser.add_argument("--standardize", default=False, action="store_true",
                    help="apply standardization to data before inference")
parser.add_argument("--dataset", default="dataset/Abilene/processed.pth", type=str,
                    help="folder from which to read the training and testing data")
parser.add_argument("--save_filename", type=str, help="name to use for the file containing the results",
                    default="contextual_indbias.pth")
opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.cpu:
    device = torch.device("cpu")
else:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{opt.device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
# device = "cpu"

epochs = opt.epochs

if opt.loss_func in ["MSE", "RMSE"]:
    criterion = nn.MSELoss()
if opt.loss_func == "L1":
    criterion = nn.SmoothL1Loss()
if opt.loss_func not in ["MSE", "RMSE", "MAPE", "MAE", "L1"]:
    raise NotImplemented("loss function not implemented")
if opt.optim not in ["Adam", "SGD"]:
    raise NotImplemented("Optimizer not implemented. Only Adam and SGD are available.")

data = torch.load(opt.dataset)  # to obtain metadata necessary for plots and other auxiliary things


@dataclass
class Metrics:
    y: torch.Tensor
    y_hat: torch.Tensor
    alphas: torch.Tensor
    MRE: float
    MSE: float
    MAE: float
    RSE: float


def compute_loss(y, y_hat):
    if opt.loss_func == "MSE":
        return criterion(y_hat, y)

    elif opt.loss_func == "RMSE":
        return torch.sqrt(criterion(y_hat, y))

    elif opt.loss_func == "MAPE":
        return torch.mean(torch.abs((y - y_hat) / y)) * 100

    elif opt.loss_func == "MAE":
        return torch.mean((y_hat - y).abs())

    elif opt.loss_func == "L1":
        return criterion(y_hat, y)


def train(train_loader, val_loader, model):
    if opt.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.sched_step, gamma=0.1)
    elif opt.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)
        sched_lambda = lambda epoch: 0.99 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_lambda)
    dataset_std = train_loader.dataset.std
    dataset_mean = train_loader.dataset.mean
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch}/{epochs}")
        loss_count = 0
        y_gt = None  # ground truth link values
        y_hat = None  # predicted link values
        alphas = []
        for sample in tqdm(train_loader):
            sample = sample.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, batch_alphas = model(
                sample
            )
            y = sample.y.to(device)
            batch_size = sample.batch.max() + 1
            num_nodes = train_loader.dataset.num_nodes
            if epoch == epochs - 1 and "wadi" not in opt.dataset.lower():  # only do this in the last epoch to reduce memory footprint
                if num_nodes * (num_nodes - 1) * batch_size != sample.edge_index.shape[1]:
                    # Not a full mesh topology
                    alphas.append(batch_alphas.view(batch_size, -1))
                else:
                    # Full mesh topology
                    zeros = torch.zeros(batch_size, num_nodes, num_nodes).to(device)
                    mask = ~(torch.eye(num_nodes) == 1).repeat(batch_size, 1, 1).to(device)
                    zeros[mask] += batch_alphas.squeeze(1)
                    zeros = zeros.permute(0, 2, 1)
                    alphas.append(zeros.detach().cpu())
            loss = compute_loss(y, pred)
            if y_gt is None:
                y_gt = y.detach().cpu().view(batch_size, -1).T
                y_hat = pred.detach().cpu().view(batch_size, -1).T
            else:
                y_gt = torch.cat((y_gt, y.detach().cpu().view(batch_size, -1).T), dim=1)
                y_hat = torch.cat((y_hat, pred.detach().cpu().view(batch_size, -1).T), dim=1)
            loss_count += loss
            loss.backward()
            optimizer.step()
        log_weights(tb, model, epoch)  # log model weights, biases and grads into tensorboard
        if opt.standardize:
            y_gt = y_gt * dataset_std.view(num_nodes, 1) + dataset_mean.view(num_nodes, 1)
            y_hat = y_hat * dataset_std.view(num_nodes, 1) + dataset_mean.view(num_nodes, 1)
        if opt.log:
            y_gt = torch.exp(y_gt) - 1
            y_hat = torch.exp(y_hat) - 1
        y_gt = y_gt.squeeze()
        y_hat = y_hat.squeeze()
        scheduler.step()
        metrics = Metrics(
            y=y_gt,
            y_hat=y_hat,
            alphas=torch.cat(alphas) if epoch == epochs - 1 and "wadi" not in opt.dataset.lower() else torch.tensor(alphas),
            MRE=(((y_gt - y_hat).abs() / torch.clamp(y_gt, min=1e-4)) * 100).mean(dim=1),
            MSE=((y_gt - y_hat) ** 2).mean(dim=1),
            MAE=(y_gt - y_hat).abs().mean(dim=1),
            RSE=(((y_gt - y_hat) ** 2).mean(dim=1) / ((y_gt - y_gt.mean()) ** 2).mean(dim=1))
        )
        log_dict = {
            "train/loss": loss_count / len(train_loader),
            "train/MRE": metrics.MRE.mean(),
            "train/MAE": metrics.MAE.mean(),
            "train/MSE": metrics.MSE.mean(),
            "train/RSE": metrics.RSE.mean(),
            "train/predictions": wandb.Image(plot_results(y=y_gt, y_hat=y_hat)),
            "epoch": epoch
        }
        metrics_val, log_val = test(val_loader, model, mode="val")
        log_dict.update(log_val)
        if opt.wandb:
            wandb.log(log_dict)
    return metrics, metrics_val


def test(loader, model, mode="test"):
    loss_count = 0
    loss_ticks = 0
    y_gt = None  # ground truth link values
    y_hat = None  # predicted link values
    dataset_std = loader.dataset.std
    dataset_mean = loader.dataset.mean
    model.eval()
    with torch.no_grad():
        alphas = []
        for sample in tqdm(loader):
            sample = sample.to(device)
            pred, batch_alphas = model(
                sample
            )
            y = sample.y
            batch_size = sample.batch.max() + 1
            num_nodes = loader.dataset.num_nodes
            if mode == "test":
                if num_nodes * (num_nodes - 1) * batch_size != sample.edge_index.shape[1]:
                    alphas.append(batch_alphas.view(batch_size, -1))
                else:
                    # Without a full-mesh topology, this does not work
                    zeros = torch.zeros(batch_size, num_nodes, num_nodes).to(device)
                    mask = ~(torch.eye(num_nodes) == 1).repeat(batch_size, 1, 1).to(device)
                    zeros[mask] += batch_alphas.squeeze(1)
                    zeros = zeros.permute(0, 2, 1)
                    alphas.append(zeros.detach().cpu())
            loss_count += compute_loss(y, pred)
            if y_gt is None:
                y_gt = y.detach().cpu().view(batch_size, -1).T
                y_hat = pred.detach().cpu().view(batch_size, -1).T
            else:
                y_gt = torch.cat((y_gt, y.detach().cpu().view(batch_size, -1).T), dim=1)
                y_hat = torch.cat((y_hat, pred.detach().cpu().view(batch_size, -1).T), dim=1)
            loss_ticks += 1
        if opt.standardize:
            y_gt = y_gt * dataset_std.view(num_nodes, 1) + dataset_mean.view(num_nodes, 1)
            y_hat = y_hat * dataset_std.view(num_nodes, 1) + dataset_mean.view(num_nodes, 1)
        if opt.log:
            y_gt = torch.exp(y_gt) - 1
            y_hat = torch.exp(y_hat) - 1
        y_gt = y_gt.squeeze()
        y_hat = y_hat.squeeze()
        metrics = Metrics(
            y=y_gt,
            y_hat=y_hat,
            alphas=torch.cat(alphas) if mode == "test" else torch.tensor(alphas),
            MRE=(((y_gt - y_hat).abs() / torch.clamp(y_gt, min=1e-4)) * 100).mean(dim=1),
            MSE=((y_gt - y_hat) ** 2).mean(dim=1),
            MAE=(y_gt - y_hat).abs().mean(dim=1),
            RSE=(((y_gt - y_hat) ** 2).mean(dim=1) / ((y_gt - y_gt.mean()) ** 2).mean(dim=1))
        )
        log_dict = {
            f"{mode}/loss": loss_count / len(loader),
            f"{mode}/MRE": metrics.MRE.mean(),
            f"{mode}/MAE": metrics.MAE.mean(),
            f"{mode}/MSE": metrics.MSE.mean(),
            f"{mode}/RSE": metrics.RSE.mean(),
            f"{mode}/predictions": wandb.Image(plot_results(y=y_gt, y_hat=y_hat))
        }
        return metrics, log_dict


if __name__ == "__main__":
    if opt.wandb:
        wandb.init(
            project="context-indbias",
            entity="hlatif",
            name=f"{opt.model} "
                 f"{'WaDi ' if 'wadi' in opt.dataset.lower() else ''}"
                 f"{'Abilene ' if 'abilene' in opt.dataset.lower() else ''}"
                 f"WS={opt.window_size} "
                 f"LR={opt.learning_rate} "
                 f"loss={opt.loss_func} "
                 f"log={'True' if opt.log else 'False'} "
                 f"standardize={'True' if opt.standardize else 'False'}",
            sync_tensorboard=True
        )
    tb = SummaryWriter(log_dir="logs")
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    config = dict(
        window_size=opt.window_size,
        val_ratio=0.1,
        test_ratio=0.5,
        log=opt.log,
        standardize=opt.standardize
    )
    if "abilene" in opt.dataset.lower():
        train_dataset, val_dataset, test_dataset = read_data(
            opt.dataset,
            config
        )
    else:
        train_dataset, val_dataset, test_dataset = read_wadi_data(
            opt.dataset,
            config
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        pin_memory=True
    )
    attrs = dict(
        num_paths=train_dataset.num_nodes,
        hidden_dimension=128,
        dropout=opt.dropout,
        window_size=opt.window_size,
        temperature=opt.att_temp
    )
    model = GAT_AD(**attrs).to(device)
    if opt.wandb:
        wandb.watch(
            model,
            log="all",
            log_freq=200,
            log_graph=True
        )
    print("Training...")
    train_metrics, val_metrics = train(train_loader, val_loader, model)
    print("Testing...")
    test_metrics, test_dict = test(test_loader, model, mode="test")
    if opt.wandb:
        wandb.log(test_dict)
    torch.save(
        dict(
            train=train_metrics,
            val=val_metrics,
            test=test_metrics,
            model_weights=model.state_dict(),
            metadata=dict(
                **opt.__dict__,
                git_version=git.Repo(search_parent_directories=True).head.object.hexsha
            )
        ),
        opt.save_filename
    )
