import argparse
import csv
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from urllib.request import urlopen

import numpy as np
import torch
from transformers import AutoModelForCausalLM


DATASETS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}

SPLIT_CONFIG = {
    "ETTh1": {"train": (0, 8640), "valid": (8640, 11520), "test": (11520, 14400)},
    "ETTh2": {"train": (0, 8640), "valid": (8640, 11520), "test": (11520, 14400)},
    "ETTm1": {"train": (0, 34560), "valid": (34560, 46080), "test": (46080, 57600)},
    "ETTm2": {"train": (0, 34560), "valid": (34560, 46080), "test": (46080, 57600)},
}

DEFAULT_MODEL_ID = "thuml/timer-base-84m"
DEFAULT_CONTEXT_LENGTH = 672
DEFAULT_PREDICTION_LENGTHS = (96, 192, 336, 720)


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


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_dataset(url: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    with urlopen(url) as response:
        text = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError(f"CSV has no header: {url}")

    timestamp_col = "date"
    target_columns = [c for c in reader.fieldnames if c != timestamp_col]
    values: Dict[str, List[float]] = {c: [] for c in target_columns}

    for row in reader:
        for col in target_columns:
            values[col].append(float(row[col]))

    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in values.items()}
    return arrays, target_columns


def build_windows(
    series: np.ndarray,
    start: int,
    end: int,
    context_length: int,
    horizon: int,
    stride: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    contexts: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for t in range(start, end - horizon + 1, stride):
        left = t - context_length
        if left < 0:
            continue
        contexts.append(series[left:t])
        targets.append(series[t : t + horizon])

    if not targets:
        return contexts, np.empty((0, horizon), dtype=np.float32)
    return contexts, np.stack(targets).astype(np.float32)


def decode_prediction(output: torch.Tensor, horizon: int) -> np.ndarray:
    if output.ndim != 2:
        raise ValueError(f"Expected 2D output, got shape {tuple(output.shape)}")
    if output.shape[1] < horizon:
        raise ValueError(
            f"Model output shorter than horizon: shape={tuple(output.shape)}, horizon={horizon}"
        )
    return output[:, -horizon:].detach().cpu().numpy().astype(np.float32)


def forecast_batches(
    model: AutoModelForCausalLM,
    contexts: Sequence[np.ndarray],
    horizon: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if not contexts:
        return np.empty((0, horizon), dtype=np.float32)

    preds: List[np.ndarray] = []
    for i in range(0, len(contexts), batch_size):
        batch = np.stack(contexts[i : i + batch_size]).astype(np.float32)
        seqs = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            output = model.generate(seqs, max_new_tokens=horizon)
        preds.append(decode_prediction(output, horizon))

    return np.concatenate(preds, axis=0)


def evaluate_dataset(
    model: AutoModelForCausalLM,
    dataset_name: str,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    stride: int,
    device: torch.device,
) -> EvalResult:
    series_map, target_columns = load_dataset(DATASETS[dataset_name])
    train_start, train_end = SPLIT_CONFIG[dataset_name]["train"]
    test_start, test_end = SPLIT_CONFIG[dataset_name]["test"]

    per_series_mse: Dict[str, float] = {}
    per_series_mae: Dict[str, float] = {}
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_windows = 0

    for col in target_columns:
        raw_series = series_map[col]
        train_values = raw_series[train_start:train_end]
        mean = float(np.mean(train_values))
        std = float(np.std(train_values))
        std = std if std > 1e-6 else 1.0
        series = (raw_series - mean) / std

        contexts, targets = build_windows(
            series=series,
            start=test_start,
            end=test_end,
            context_length=context_length,
            horizon=prediction_length,
            stride=stride,
        )
        preds = forecast_batches(
            model=model,
            contexts=contexts,
            horizon=prediction_length,
            batch_size=batch_size,
            device=device,
        )

        if preds.shape != targets.shape:
            raise ValueError(
                f"Prediction/target mismatch for {dataset_name}:{col} -> "
                f"{preds.shape} vs {targets.shape}"
            )

        mse = float(np.mean((preds - targets) ** 2)) if preds.size else math.nan
        mae = float(np.mean(np.abs(preds - targets))) if preds.size else math.nan
        per_series_mse[col] = mse
        per_series_mae[col] = mae
        total_windows += len(contexts)
        all_preds.append(preds)
        all_targets.append(targets)

    flat_preds = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0, 0))
    flat_targets = (
        np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, 0))
    )
    overall_mse = (
        float(np.mean((flat_preds - flat_targets) ** 2)) if flat_preds.size else math.nan
    )
    overall_mae = (
        float(np.mean(np.abs(flat_preds - flat_targets))) if flat_preds.size else math.nan
    )

    return EvalResult(
        dataset=dataset_name,
        prediction_length=prediction_length,
        windows=total_windows,
        series_count=len(target_columns),
        overall_mse=overall_mse,
        overall_mae=overall_mae,
        per_series_mse=per_series_mse,
        per_series_mae=per_series_mae,
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
        description="Timer-XL zero-shot evaluation on ETT datasets."
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument(
        "--prediction-lengths",
        type=str,
        default=",".join(map(str, DEFAULT_PREDICTION_LENGTHS)),
        help="Comma-separated forecast lengths, e.g. 96,192,336,720",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="ETTh1,ETTh2,ETTm1,ETTm2",
        help="Comma-separated dataset names",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Window step. 0 means stride=prediction_length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    args = parser.parse_args()

    prediction_lengths = parse_int_list(args.prediction_lengths)
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for dataset_name in dataset_names:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    all_results: List[EvalResult] = []
    for dataset_name in dataset_names:
        for pred_len in prediction_lengths:
            stride = args.stride if args.stride > 0 else pred_len
            print(
                f"\nRunning {dataset_name} with context={args.context_length}, "
                f"prediction={pred_len}, stride={stride}"
            )
            result = evaluate_dataset(
                model=model,
                dataset_name=dataset_name,
                context_length=args.context_length,
                prediction_length=pred_len,
                batch_size=args.batch_size,
                stride=stride,
                device=device,
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
