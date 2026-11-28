"""Task 1: CIFAR-10 classification with an MLP and CNN."""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CIFAR10MLP(nn.Module):
    """Flattens 3x32x32 CIFAR-10 images (3072 features) through >=2 hidden layers"""

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 2:
            raise ValueError("Task 1 requires at least 2 hidden layers in the MLP.")

        layers: List[nn.Module] = [nn.Flatten()]
        in_features = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = h_dim

        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def peak_memory_mib(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    # Fallback to process RSS on CPU (Linux /proc/self/status)
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            if line.startswith("VmHWM:"):
                kb = float(line.split()[1])
                return kb / 1024.0
    return None


def load_cifar10(data_dir: str, train: bool) -> datasets.CIFAR10:
    # Standard CIFAR-10 mean and std per channel
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * y_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return running_loss / total, correct / total


def visualize_first_layer_weights(model: nn.Module, out_path: Path, n_neurons: int = 16) -> None:
    """Visualizes rows of the first Linear layer reshaped back to (32, 32, 3) images."""
    first_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        return

    # Shape: (out_features, 3072) -> (out_features, 3, 32, 32) -> (out_features, 32, 32, 3)
    weights = first_linear.weight.detach().cpu().numpy()
    n_show = min(n_neurons, weights.shape[0])
    scale = np.abs(weights[:n_show]).max()
    cols = 4
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx < n_show:
            w_img = weights[idx].reshape(3, 32, 32).transpose(1, 2, 0)
            # Use one scale for all tiles. Zero maps to mid-gray.
            rgb = 0.5 + 0.5 * w_img / scale if scale > 0 else np.full_like(w_img, 0.5)
            ax.imshow(rgb)
            ax.set_title(f"Unit {idx}", fontsize=9)
        ax.axis("off")

    plt.suptitle("MLP First-Layer Weights (32×32 RGB, Zero Is Mid-Gray)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def run_cv_for_config(
    config: dict,
    train_dataset: datasets.CIFAR10,
    folds: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    criterion = nn.CrossEntropyLoss()

    fold_val_accs: List[float] = []
    fold_times: List[float] = []
    hidden_dims = tuple(config["hidden_dims"])

    splits = skf.split(np.zeros(len(train_dataset)), train_dataset.targets)
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        set_seed(seed + fold_idx)

        train_loader = DataLoader(
            Subset(train_dataset, train_idx),
            batch_size=config["batch_size"],
            shuffle=True,
            # Batch normalization needs at least two examples per training batch.
            drop_last=len(train_idx) % config["batch_size"] == 1,
        )
        val_loader = DataLoader(
            Subset(train_dataset, val_idx),
            batch_size=config["batch_size"],
            shuffle=False,
        )

        model = CIFAR10MLP(
            hidden_dims=hidden_dims,
            num_classes=10,
            dropout=config["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        start_t = time.perf_counter()
        for _ in range(epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
        fold_times.append(time.perf_counter() - start_t)

        # Score the last epoch, which is also the one used in the final fit.
        _, val_acc = evaluate(model, val_loader, criterion, device)
        fold_val_accs.append(val_acc)

    total_params, _ = count_parameters(
        CIFAR10MLP(hidden_dims=hidden_dims, dropout=config["dropout"])
    )

    return {
        "name": config["name"],
        "hidden_dims": list(hidden_dims),
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "dropout": config["dropout"],
        "weight_decay": config["weight_decay"],
        "params": total_params,
        "cv_val_acc_mean": float(np.mean(fold_val_accs)),
        "cv_val_acc_std": float(np.std(fold_val_accs)),
        "fold_val_accuracies": fold_val_accs,
        "mean_fold_train_time_sec": float(np.mean(fold_times)),
    }


def search_mlp_hyperparameters(args: argparse.Namespace, out_dir: Path, device: torch.device) -> dict:
    train_dataset = load_cifar10(args.data_dir, train=True)
    # Hyperparameter search candidates (for the appendix comparison table)
    search_space = [
        {"name": "mlp_1024_512_lr1e-3", "hidden_dims": (1024, 512), "lr": 1e-3, "batch_size": 128, "dropout": 0.2, "weight_decay": 1e-4},
        {"name": "mlp_1024_512_lr3e-4", "hidden_dims": (1024, 512), "lr": 3e-4, "batch_size": 128, "dropout": 0.2, "weight_decay": 1e-4},
        {"name": "mlp_2048_1024_512", "hidden_dims": (2048, 1024, 512), "lr": 5e-4, "batch_size": 128, "dropout": 0.3, "weight_decay": 1e-4},
        {"name": "mlp_512_256_bs256", "hidden_dims": (512, 256), "lr": 1e-3, "batch_size": 256, "dropout": 0.1, "weight_decay": 0.0},
    ]

    results = []
    for cfg in search_space:
        print(f"Running CV for {cfg['name']}...")
        res = run_cv_for_config(cfg, train_dataset, args.folds, args.epochs, args.seed, device)
        results.append(res)
        print(
            f"  -> CV Val Acc: {res['cv_val_acc_mean']:.4f} +/- {res['cv_val_acc_std']:.4f} | "
            f"Params: {res['params']:,} | Avg Fold Train Time: {res['mean_fold_train_time_sec']:.1f}s"
        )

    # Save hyperparameter search results for your appendix
    csv_path = out_dir / "mlp_hyperparam_search.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    return max(results, key=lambda result: result["cv_val_acc_mean"])


def fit_final_mlp(
    args: argparse.Namespace, out_dir: Path, device: torch.device, best_res: dict
) -> None:
    """Train the selected configuration and evaluate it on the held-out test set."""
    set_seed(args.seed)
    train_dataset = load_cifar10(args.data_dir, train=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_model = CIFAR10MLP(
        hidden_dims=tuple(best_res["hidden_dims"]),
        num_classes=10,
        dropout=best_res["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        best_model.parameters(),
        lr=best_res["lr"],
        weight_decay=best_res["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    full_train_loader = DataLoader(train_dataset, batch_size=best_res["batch_size"], shuffle=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_train_start = time.perf_counter()
    for _ in range(args.epochs):
        train_one_epoch(best_model, full_train_loader, criterion, optimizer, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_train_time = time.perf_counter() - final_train_start

    # Capture training memory before loading any test data or plotting weights.
    final_peak_memory = peak_memory_mib(device)
    test_dataset = load_cifar10(args.data_dir, train=False)
    test_loader = DataLoader(test_dataset, batch_size=best_res["batch_size"], shuffle=False)
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)
    visualize_first_layer_weights(best_model, out_dir / "mlp_first_layer_weights.png")

    summary = {
        "best_config": best_res,
        "folds": args.folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "final_train_time_sec": final_train_time,
        "final_train_sec_per_epoch": final_train_time / args.epochs,
        "final_peak_memory_mib": final_peak_memory,
        "memory_metric": "CUDA peak allocated memory" if device.type == "cuda" else "CPU process peak RSS",
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    (out_dir / "mlp_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest config: {best_res['name']} | Test Acc: {test_acc:.4f} | Saved to {out_dir}")


class CIFAR10CNN(nn.Module):
    """Three 3×3 convolutions and a fully connected layer with 10 outputs."""

    def __init__(self, channels: tuple[int, int, int], dropout: float) -> None:
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        # The three pooling layers halve the image size: 32 -> 16 -> 8 -> 4.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4 * 4, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


# Change one setting at a time relative to the baseline.
CNN_BASE_CONFIG = {
    "name": "cnn_baseline",
    "channels": (16, 32, 64),
    "lr": 1e-3,
    "batch_size": 128,
    "dropout": 0.2,
    "weight_decay": 1e-4,
}

CNN_SEARCH_SPACE = [
    CNN_BASE_CONFIG,
    {**CNN_BASE_CONFIG, "name": "cnn_lower_lr", "lr": 3e-4},
    {**CNN_BASE_CONFIG, "name": "cnn_wider", "channels": (24, 48, 96)},
    {**CNN_BASE_CONFIG, "name": "cnn_more_dropout", "dropout": 0.4},
]


def fit_cnn(
    config: dict, train_loader: DataLoader, epochs: int, device: torch.device
) -> tuple[CIFAR10CNN, float]:
    """Train a fresh model and time the training loop."""
    model = CIFAR10CNN(tuple(config["channels"]), config["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, time.perf_counter() - start


def search_cnn_hyperparameters(
    args: argparse.Namespace, out_dir: Path, device: torch.device
) -> dict:
    train_dataset = load_cifar10(args.data_dir, train=True)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.zeros(len(train_dataset)), train_dataset.targets))
    criterion = nn.CrossEntropyLoss()
    results = []

    for config in CNN_SEARCH_SPACE:
        print(f"Running CV for {config['name']}...", flush=True)
        accuracies = []
        times = []
        batch_size = config["batch_size"]

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            set_seed(args.seed + fold_idx)
            train_loader = DataLoader(
                Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                Subset(train_dataset, val_idx), batch_size=batch_size, shuffle=False
            )
            model, elapsed = fit_cnn(config, train_loader, args.epochs, device)
            _, accuracy = evaluate(model, val_loader, criterion, device)
            _, trainable_params = count_parameters(model)
            del model
            accuracies.append(accuracy)
            times.append(elapsed)
            print(
                f"  Fold {fold_idx + 1}/{args.folds}: "
                f"validation accuracy {accuracy:.2%}, training {elapsed:.1f}s",
                flush=True,
            )

        result = {
            **config,
            "params": trainable_params,
            "cv_val_acc_mean": float(np.mean(accuracies)),
            "cv_val_acc_std": float(np.std(accuracies)),
            "fold_val_accuracies": accuracies,
            "mean_fold_train_time_sec": float(np.mean(times)),
        }
        results.append(result)
        print(f"  Mean validation accuracy: {result['cv_val_acc_mean']:.2%}", flush=True)

    with (out_dir / "cnn_hyperparam_search.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    return max(results, key=lambda result: result["cv_val_acc_mean"])


def visualize_first_layer_kernels(model: CIFAR10CNN, out_path: Path) -> None:
    """Show up to 16 first-layer RGB kernels on a shared scale centered at zero."""
    kernels = model.features[0].weight.detach().cpu().numpy()[:16]
    scale = np.abs(kernels).max()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for index, ax in enumerate(axes.flat):
        if index < len(kernels):
            kernel = kernels[index].transpose(1, 2, 0)
            rgb = 0.5 + 0.5 * kernel / scale if scale > 0 else np.full_like(kernel, 0.5)
            ax.imshow(rgb, interpolation="nearest")
            ax.set_title(f"Kernel {index}")
        ax.axis("off")

    fig.suptitle("CNN First-Layer Kernels (3×3 RGB, Zero Is Mid-Gray)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fit_final_cnn(
    args: argparse.Namespace, out_dir: Path, device: torch.device, best_config: dict
) -> None:
    set_seed(args.seed)
    train_dataset = load_cifar10(args.data_dir, train=True)
    batch_size = best_config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, train_time = fit_cnn(best_config, train_loader, args.epochs, device)
    # Record the training peak before loading test data or drawing the plot.
    training_peak = peak_memory_mib(device)

    test_dataset = load_cifar10(args.data_dir, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    visualize_first_layer_kernels(model, out_dir / "cnn_first_layer_kernels.png")

    summary = {
        "best_config": best_config,
        "folds": args.folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "final_train_time_sec": train_time,
        "final_train_sec_per_epoch": train_time / args.epochs,
        "final_peak_memory_mib": training_peak,
        "memory_metric": "CUDA peak allocated memory" if device.type == "cuda" else "CPU process peak RSS",
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    (out_dir / "cnn_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest config: {best_config['name']} | Test Acc: {test_acc:.4f} | Saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: MLP and CNN on CIFAR-10")
    parser.add_argument("--model", choices=("both", "mlp", "cnn"), default="both")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./outputs/task1")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per fold and final fit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--final-config", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires a CUDA-enabled PyTorch and an available GPU")

    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    if args.model == "both":
        if args.final_config is not None:
            parser.error("--final-config requires a single model")
        for model_name in ("mlp", "cnn"):
            subprocess.run([*command, "--model", model_name], check=True)
        return

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "mlp":
        search = search_mlp_hyperparameters
        fit_final = fit_final_mlp
    else:
        search = search_cnn_hyperparameters
        fit_final = fit_final_cnn

    if args.final_config is not None:
        print(f"Training the selected {args.model.upper()} on {device}.", flush=True)
        fit_final(args, out_dir, device, json.loads(args.final_config))
        return

    print(f"Running {args.model.upper()} cross-validation on {device}.", flush=True)
    best_config = search(args, out_dir, device)
    # Start the final fit in a fresh process for peak memory measurement.
    sys.stdout.flush()
    os.execv(sys.executable, command + ["--final-config", json.dumps(best_config)])


if __name__ == "__main__":
    main()
