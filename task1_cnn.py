"""Task 1 (CNN): CIFAR-10 classification with cross-validation and kernel plots."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset

from task1_mlp_cnn import (
    count_parameters,
    evaluate,
    load_cifar10,
    peak_memory_mib,
    set_seed,
    train_one_epoch,
)


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
BASE_CONFIG = {
    "name": "cnn_baseline",
    "channels": (16, 32, 64),
    "lr": 1e-3,
    "batch_size": 128,
    "dropout": 0.2,
    "weight_decay": 1e-4,
}

SEARCH_SPACE = [
    BASE_CONFIG,
    {**BASE_CONFIG, "name": "cnn_lower_lr", "lr": 3e-4},
    {**BASE_CONFIG, "name": "cnn_wider", "channels": (24, 48, 96)},
    {**BASE_CONFIG, "name": "cnn_more_dropout", "dropout": 0.4},
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


def search_hyperparameters(
    args: argparse.Namespace, out_dir: Path, device: torch.device
) -> dict:
    train_dataset = load_cifar10(args.data_dir, train=True)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.zeros(len(train_dataset)), train_dataset.targets))
    criterion = nn.CrossEntropyLoss()
    results = []

    for config in SEARCH_SPACE:
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


def fit_final_model(
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
    parser = argparse.ArgumentParser(description="Task 1 CNN on CIFAR-10")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./outputs/task1_cnn")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
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

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.final_config is not None:
        print(f"Training the selected CNN on {device}.", flush=True)
        fit_final_model(args, out_dir, device, json.loads(args.final_config))
        return

    print(f"Running CNN cross-validation on {device}.", flush=True)
    best_config = search_hyperparameters(args, out_dir, device)
    # Match the MLP: measure final training in a fresh process after the search.
    sys.stdout.flush()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    os.execv(sys.executable, command + ["--final-config", json.dumps(best_config)])


if __name__ == "__main__":
    main()
