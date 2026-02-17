import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import urlopen

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
TIMECMA_ROOT = PROJECT_ROOT / "TimeCMA"
if str(TIMECMA_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMECMA_ROOT))
if str(TIMECMA_ROOT / "storage") not in sys.path:
    sys.path.insert(0, str(TIMECMA_ROOT / "storage"))

from data_provider.data_loader_emb import (  # noqa: E402
    Dataset_ETT_hour as TrainDatasetETTHour,
    Dataset_ETT_minute as TrainDatasetETTMinute,
)
from data_provider.data_loader_save import (  # noqa: E402
    Dataset_ETT_hour as SaveDatasetETTHour,
    Dataset_ETT_minute as SaveDatasetETTMinute,
)
from models.TimeCMA import Dual  # noqa: E402
from storage.gen_prompt_emb import GenPromptEmb  # noqa: E402
from utils.metrics import MAE, MSE, metric  # noqa: E402


DATASETS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}

DEFAULT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
DEFAULT_PREDICTION_LENGTHS = (96, 192, 336, 720)


@dataclass(frozen=True)
class RunConfig:
    learning_rate: float
    channel: int
    e_layer: int
    d_layer: int
    dropout_n: float
    head: int = 8


@dataclass(frozen=True)
class DatasetConfig:
    seq_len: int
    batch_size: int
    num_nodes: int
    seed: int
    epochs: int
    run_by_pred_len: Dict[int, RunConfig]


@dataclass
class EvalResult:
    dataset: str
    prediction_length: int
    windows: int
    series_count: int
    overall_mse: float
    overall_mae: float
    per_series_mse: Dict[str, float]
    per_series_mae: Dict[str, float]


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "ETTh1": DatasetConfig(
        seq_len=96,
        batch_size=16,
        num_nodes=7,
        seed=2024,
        epochs=999,
        run_by_pred_len={
            96: RunConfig(1e-4, 64, 1, 2, 0.7),
            192: RunConfig(1e-4, 64, 1, 2, 0.7),
            336: RunConfig(1e-4, 64, 1, 2, 0.7),
            720: RunConfig(1e-4, 32, 2, 2, 0.8),
        },
    ),
    "ETTh2": DatasetConfig(
        seq_len=96,
        batch_size=16,
        num_nodes=7,
        seed=2024,
        epochs=999,
        run_by_pred_len={
            96: RunConfig(1e-4, 64, 2, 2, 0.3),
            192: RunConfig(1e-4, 64, 2, 2, 0.3),
            336: RunConfig(1e-4, 64, 2, 2, 0.3),
            720: RunConfig(1e-4, 64, 2, 2, 0.3),
        },
    ),
    "ETTm1": DatasetConfig(
        seq_len=96,
        batch_size=16,
        num_nodes=7,
        seed=2024,
        epochs=999,
        run_by_pred_len={
            96: RunConfig(1e-4, 64, 2, 2, 0.5),
            192: RunConfig(1e-4, 64, 2, 2, 0.5),
            336: RunConfig(1e-4, 64, 2, 2, 0.5),
            720: RunConfig(1e-4, 64, 2, 2, 0.7),
        },
    ),
    "ETTm2": DatasetConfig(
        seq_len=96,
        batch_size=16,
        num_nodes=7,
        seed=2024,
        epochs=999,
        run_by_pred_len={
            96: RunConfig(1e-4, 64, 2, 2, 0.3),
            192: RunConfig(1e-4, 64, 2, 2, 0.3),
            336: RunConfig(1e-4, 64, 2, 2, 0.3),
            720: RunConfig(1e-4, 64, 2, 2, 0.3),
        },
    ),
}


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def seed_it(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def load_series_names(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if not header:
        return []
    return [col for col in header if col != "date"]


def maybe_download_csv(dataset_name: str, dataset_url: str, data_root: Path, force: bool) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    out_path = data_root / f"{dataset_name}.csv"
    if out_path.exists() and not force:
        return out_path
    with urlopen(dataset_url) as resp:
        text = resp.read()
    out_path.write_bytes(text)
    return out_path


def get_train_dataset_class(dataset_name: str):
    if dataset_name in {"ETTh1", "ETTh2"}:
        return TrainDatasetETTHour
    if dataset_name in {"ETTm1", "ETTm2"}:
        return TrainDatasetETTMinute
    raise ValueError(f"Unsupported dataset for TimeCMA runner: {dataset_name}")


def get_save_dataset_class(dataset_name: str):
    if dataset_name in {"ETTh1", "ETTh2"}:
        return SaveDatasetETTHour
    if dataset_name in {"ETTm1", "ETTm2"}:
        return SaveDatasetETTMinute
    raise ValueError(f"Unsupported dataset for TimeCMA runner: {dataset_name}")


def ensure_embeddings(
    dataset_name: str,
    seq_len: int,
    embed_output_len: int,
    data_root: Path,
    embedding_root: Path,
    model_name: str,
    d_model: int,
    llm_layers: int,
    device: torch.device,
    num_workers: int,
    force_regenerate: bool,
) -> None:
    save_dataset_class = get_save_dataset_class(dataset_name)
    generator = GenPromptEmb(
        data_path=dataset_name,
        model_name=model_name,
        device=str(device),
        input_len=seq_len,
        d_model=d_model,
        layer=llm_layers,
    ).to(device)

    for split in ("train", "val", "test"):
        split_dir = embedding_root / dataset_name / split
        split_dir.mkdir(parents=True, exist_ok=True)

        dataset = save_dataset_class(
            root_path=str(data_root.resolve()),
            flag=split,
            size=[seq_len, 0, embed_output_len],
            data_path=dataset_name,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

        for idx, (x, _y, x_mark, _y_mark) in enumerate(loader):
            file_path = split_dir / f"{idx}.h5"
            if file_path.exists() and not force_regenerate:
                continue
            emb = generator.generate_embeddings(x.to(device), x_mark.to(device))
            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("embeddings", data=emb.detach().cpu().numpy())


def build_loaders(
    dataset_name: str,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    data_root: Path,
    embedding_root: Path,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_class = get_train_dataset_class(dataset_name)
    train_set = data_class(
        root_path=str(data_root.resolve()),
        flag="train",
        size=[seq_len, 0, pred_len],
        data_path=dataset_name,
    )
    val_set = data_class(
        root_path=str(data_root.resolve()),
        flag="val",
        size=[seq_len, 0, pred_len],
        data_path=dataset_name,
    )
    test_set = data_class(
        root_path=str(data_root.resolve()),
        flag="test",
        size=[seq_len, 0, pred_len],
        data_path=dataset_name,
    )

    train_set.embed_path = str((embedding_root / dataset_name / "train").resolve()) + "/"
    val_set.embed_path = str((embedding_root / dataset_name / "val").resolve()) + "/"
    test_set.embed_path = str((embedding_root / dataset_name / "test").resolve()) + "/"

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def evaluate_model(
    model: Dual,
    test_loader: DataLoader,
    device: torch.device,
    series_names: Sequence[str],
    dataset_name: str,
    pred_len: int,
) -> EvalResult:
    model.eval()
    preds_list: List[torch.Tensor] = []
    truth_list: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y, x_mark, _y_mark, embeddings in test_loader:
            testx = torch.as_tensor(x, device=device)
            testy = torch.as_tensor(y, device=device)
            testx_mark = torch.as_tensor(x_mark, device=device)
            test_embedding = torch.as_tensor(embeddings, device=device)
            preds = model(testx, testx_mark, test_embedding)
            preds_list.append(preds)
            truth_list.append(testy)

    if not preds_list:
        raise RuntimeError("No test windows available. Check dataset sizes and batch size.")

    test_pred = torch.cat(preds_list, dim=0)
    test_true = torch.cat(truth_list, dim=0)
    overall_mse = MSE(test_pred, test_true).item()
    overall_mae = MAE(test_pred, test_true).item()

    per_series_mse: Dict[str, float] = {}
    per_series_mae: Dict[str, float] = {}
    n_series = int(test_pred.shape[-1])

    for i in range(n_series):
        name = series_names[i] if i < len(series_names) else f"series_{i}"
        mse_i, mae_i = metric(test_pred[:, :, i], test_true[:, :, i])
        per_series_mse[name] = float(mse_i)
        per_series_mae[name] = float(mae_i)

    return EvalResult(
        dataset=dataset_name,
        prediction_length=pred_len,
        windows=int(test_pred.shape[0]),
        series_count=n_series,
        overall_mse=float(overall_mse),
        overall_mae=float(overall_mae),
        per_series_mse=per_series_mse,
        per_series_mae=per_series_mae,
    )


def run_single(
    dataset_name: str,
    pred_len: int,
    dataset_cfg: DatasetConfig,
    run_cfg: RunConfig,
    data_root: Path,
    embedding_root: Path,
    checkpoint_root: Path,
    device: torch.device,
    num_workers: int,
    es_patience: int,
    d_llm: int,
    weight_decay: float,
    epochs: int,
    batch_size_override: Optional[int],
    force_train: bool,
    reuse_checkpoints: bool,
) -> EvalResult:
    batch_size = batch_size_override if batch_size_override else dataset_cfg.batch_size
    seed_it(dataset_cfg.seed)
    series_names = load_series_names(data_root / f"{dataset_name}.csv")

    train_loader, val_loader, test_loader = build_loaders(
        dataset_name=dataset_name,
        seq_len=dataset_cfg.seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        data_root=data_root,
        embedding_root=embedding_root,
        num_workers=num_workers,
    )

    run_dir = checkpoint_root / dataset_name / (
        f"pred{pred_len}_c{run_cfg.channel}_el{run_cfg.e_layer}_"
        f"dl{run_cfg.d_layer}_lr{run_cfg.learning_rate}_dn{run_cfg.dropout_n}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_model.pth"

    model = Dual(
        device=str(device),
        channel=run_cfg.channel,
        num_nodes=dataset_cfg.num_nodes,
        seq_len=dataset_cfg.seq_len,
        pred_len=pred_len,
        dropout_n=run_cfg.dropout_n,
        d_llm=d_llm,
        e_layer=run_cfg.e_layer,
        d_layer=run_cfg.d_layer,
        head=run_cfg.head,
    ).to(device)

    should_train = force_train or not (reuse_checkpoints and ckpt_path.exists())
    if should_train:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=run_cfg.learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=min(max(epochs, 1), 50),
            eta_min=1e-6,
        )
        best_val = float("inf")
        patience = 0

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: List[float] = []

            for x, y, x_mark, _y_mark, embeddings in train_loader:
                trainx = torch.as_tensor(x, device=device)
                trainy = torch.as_tensor(y, device=device)
                trainx_mark = torch.as_tensor(x_mark, device=device)
                train_embedding = torch.as_tensor(embeddings, device=device)

                optimizer.zero_grad()
                predict = model(trainx, trainx_mark, train_embedding)
                loss = MSE(predict, trainy)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

            model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for x, y, x_mark, _y_mark, embeddings in val_loader:
                    valx = torch.as_tensor(x, device=device)
                    valy = torch.as_tensor(y, device=device)
                    valx_mark = torch.as_tensor(x_mark, device=device)
                    val_embedding = torch.as_tensor(embeddings, device=device)
                    val_pred = model(valx, valx_mark, val_embedding)
                    val_loss = MSE(val_pred, valy)
                    val_losses.append(float(val_loss.item()))

            mean_train = float(np.mean(train_losses)) if train_losses else float("inf")
            mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
            print(
                f"Epoch {epoch:03d} | {dataset_name} pred={pred_len} "
                f"train_mse={mean_train:.4f} val_mse={mean_val:.4f}"
            )

            if mean_val < best_val:
                best_val = mean_val
                patience = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience += 1

            scheduler.step()
            if patience >= es_patience and epoch >= max(1, epochs // 2):
                break

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        series_names=series_names,
        dataset_name=dataset_name,
        pred_len=pred_len,
    )


def print_result(result: EvalResult) -> None:
    print(
        f"[{result.dataset}] pred={result.prediction_length} "
        f"windows={result.windows} series={result.series_count} "
        f"overall_mse={result.overall_mse:.6f} "
        f"overall_mae={result.overall_mae:.6f}"
    )
    for name in result.per_series_mse:
        print(
            f"  - {name}: mse={result.per_series_mse[name]:.6f}, "
            f"mae={result.per_series_mae[name]:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TimeCMA benchmark runner with timemoe_zeroshot.py-style output."
    )
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--prediction-lengths",
        type=str,
        default=",".join(map(str, DEFAULT_PREDICTION_LENGTHS)),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(TIMECMA_ROOT / "datasets"),
        help="Local directory for CSV files.",
    )
    parser.add_argument(
        "--embedding-root",
        type=str,
        default=str(TIMECMA_ROOT / "Embeddings"),
        help="Local directory for saved prompt embeddings.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(TIMECMA_ROOT / "logs_zeroshot"),
        help="Directory for trained checkpoints.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=-1, help="-1 uses script defaults per dataset.")
    parser.add_argument("--es-patience", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--embedding-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses script defaults.")
    parser.add_argument("--d-llm", type=int, default=768)
    parser.add_argument("--llm-layers", type=int, default=12)
    parser.add_argument("--llm-model-name", type=str, default="gpt2")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--download-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-embedding-generation", action="store_true")
    parser.add_argument("--force-regenerate-embeddings", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--reuse-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    datasets = parse_str_list(args.datasets)
    prediction_lengths = parse_int_list(args.prediction_lengths)
    if not datasets:
        raise ValueError("No datasets provided.")
    if not prediction_lengths:
        raise ValueError("No prediction lengths provided.")

    for dataset_name in datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"No TimeCMA script config found for dataset: {dataset_name}")
        for pred_len in prediction_lengths:
            if pred_len not in DATASET_CONFIGS[dataset_name].run_by_pred_len:
                raise ValueError(
                    f"No script config for dataset={dataset_name}, prediction_length={pred_len}"
                )

    data_root = Path(args.data_root).resolve()
    embedding_root = Path(args.embedding_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.download_data:
        print("Checking/downloading datasets...")
        for dataset_name in datasets:
            maybe_download_csv(
                dataset_name=dataset_name,
                dataset_url=DATASETS[dataset_name],
                data_root=data_root,
                force=args.force_download,
            )

    if not args.skip_embedding_generation:
        min_pred_len = min(prediction_lengths)
        for dataset_name in datasets:
            cfg = DATASET_CONFIGS[dataset_name]
            print(
                f"Generating embeddings for {dataset_name} "
                f"(seq_len={cfg.seq_len}, output_len={min_pred_len})..."
            )
            ensure_embeddings(
                dataset_name=dataset_name,
                seq_len=cfg.seq_len,
                embed_output_len=min_pred_len,
                data_root=data_root,
                embedding_root=embedding_root,
                model_name=args.llm_model_name,
                d_model=args.d_llm,
                llm_layers=args.llm_layers,
                device=device,
                num_workers=args.embedding_workers,
                force_regenerate=args.force_regenerate_embeddings,
            )

    all_results: List[EvalResult] = []
    for dataset_name in datasets:
        cfg = DATASET_CONFIGS[dataset_name]
        for pred_len in prediction_lengths:
            run_cfg = cfg.run_by_pred_len[pred_len]
            run_epochs = cfg.epochs if args.epochs < 0 else args.epochs
            print(
                f"\nRunning {dataset_name} with context={cfg.seq_len}, "
                f"prediction={pred_len}, epochs={run_epochs}"
            )
            result = run_single(
                dataset_name=dataset_name,
                pred_len=pred_len,
                dataset_cfg=cfg,
                run_cfg=run_cfg,
                data_root=data_root,
                embedding_root=embedding_root,
                checkpoint_root=checkpoint_root,
                device=device,
                num_workers=args.num_workers,
                es_patience=args.es_patience,
                d_llm=args.d_llm,
                weight_decay=args.weight_decay,
                epochs=run_epochs,
                batch_size_override=args.batch_size if args.batch_size > 0 else None,
                force_train=args.force_train,
                reuse_checkpoints=args.reuse_checkpoints,
            )
            print_result(result)
            all_results.append(result)

    print("\nSummary (dataset, prediction_length, overall_mse, overall_mae)")
    for result in all_results:
        print(
            f"{result.dataset},{result.prediction_length},"
            f"{result.overall_mse:.6f},{result.overall_mae:.6f}"
        )

    print("\nAveraged across prediction lengths (dataset, avg_mse, avg_mae)")
    by_dataset: Dict[str, List[EvalResult]] = {}
    for result in all_results:
        by_dataset.setdefault(result.dataset, []).append(result)
    for dataset, rows in by_dataset.items():
        avg_mse = float(np.mean([r.overall_mse for r in rows]))
        avg_mae = float(np.mean([r.overall_mae for r in rows]))
        print(f"{dataset},{avg_mse:.6f},{avg_mae:.6f}")


if __name__ == "__main__":
    main()
