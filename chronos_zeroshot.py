import argparse
import csv
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union
from urllib.request import urlopen

import numpy as np
import torch
from chronos import Chronos2Pipeline, ChronosPipeline


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

DEFAULT_MODEL_ID = "amazon/chronos-2"
DEFAULT_CONTEXT_LENGTH = 672
DEFAULT_PREDICTION_LENGTHS = (96, 192, 336, 720)

PipelineType = Union[ChronosPipeline, Chronos2Pipeline]


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


def decode_chronos_samples(prediction: torch.Tensor, horizon: int) -> np.ndarray:
    arr = prediction.detach().float().cpu().numpy().astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3:
        # ChronosPipeline: (batch, num_samples, horizon)
        arr = arr.mean(axis=1)

    if arr.ndim != 2:
        raise ValueError(f"Unexpected Chronos prediction shape: {arr.shape}")
    if arr.shape[1] < horizon:
        raise ValueError(
            f"Chronos prediction shorter than horizon: shape={arr.shape}, horizon={horizon}"
        )
    if arr.shape[1] > horizon:
        arr = arr[:, -horizon:]
    return arr


def decode_chronos2_quantiles(
    predictions: Sequence[torch.Tensor],
    quantiles: Sequence[float],
    horizon: int,
) -> np.ndarray:
    if not predictions:
        return np.empty((0, horizon), dtype=np.float32)
    if not quantiles:
        raise ValueError("Chronos-2 pipeline returned no quantile levels.")

    quantile_idx = int(np.argmin(np.abs(np.asarray(quantiles, dtype=np.float32) - 0.5)))
    rows: List[np.ndarray] = []

    for pred in predictions:
        arr = pred.detach().float().cpu().numpy().astype(np.float32)
        if arr.ndim == 3:
            # (n_variates, n_quantiles, horizon), use first variate for univariate series.
            if arr.shape[0] < 1:
                raise ValueError(f"Invalid Chronos-2 output shape: {arr.shape}")
            row = arr[0, quantile_idx, :]
        elif arr.ndim == 2:
            # (n_quantiles, horizon)
            row = arr[quantile_idx, :]
        elif arr.ndim == 1:
            row = arr
        else:
            raise ValueError(f"Unexpected Chronos-2 output shape: {arr.shape}")

        if row.shape[0] < horizon:
            raise ValueError(
                f"Chronos-2 prediction shorter than horizon: shape={row.shape}, horizon={horizon}"
            )
        if row.shape[0] > horizon:
            row = row[-horizon:]
        rows.append(row)

    return np.stack(rows).astype(np.float32)


def forecast_batches(
    pipeline: PipelineType,
    contexts: Sequence[np.ndarray],
    horizon: int,
    batch_size: int,
    num_samples: int,
) -> np.ndarray:
    if not contexts:
        return np.empty((0, horizon), dtype=np.float32)

    preds: List[np.ndarray] = []
    for i in range(0, len(contexts), batch_size):
        batch_contexts = [
            torch.from_numpy(ctx.astype(np.float32)) for ctx in contexts[i : i + batch_size]
        ]

        if isinstance(pipeline, Chronos2Pipeline):
            batch_output = pipeline.predict(
                inputs=batch_contexts,
                prediction_length=horizon,
                batch_size=len(batch_contexts),
            )
            batch_preds = decode_chronos2_quantiles(
                predictions=batch_output,
                quantiles=pipeline.quantiles,
                horizon=horizon,
            )
        else:
            batch_output = pipeline.predict(
                inputs=batch_contexts,
                prediction_length=horizon,
                num_samples=num_samples,
            )
            batch_preds = decode_chronos_samples(batch_output, horizon)

        if batch_preds.ndim != 2 or batch_preds.shape[1] != horizon:
            raise ValueError(
                f"Unexpected decoded prediction shape {batch_preds.shape} for horizon={horizon}"
            )
        preds.append(batch_preds)

    return np.concatenate(preds, axis=0)


def evaluate_dataset(
    pipeline: PipelineType,
    dataset_name: str,
    series_map: Dict[str, np.ndarray],
    target_columns: List[str],
    context_length: int,
    prediction_length: int,
    batch_size: int,
    stride: int,
    num_samples: int,
) -> EvalResult:
    test_start, test_end = SPLIT_CONFIG[dataset_name]["test"]

    per_series_mse: Dict[str, float] = {}
    per_series_mae: Dict[str, float] = {}
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_windows = 0

    for col in target_columns:
        series = series_map[col]
        contexts, targets = build_windows(
            series=series,
            start=test_start,
            end=test_end,
            context_length=context_length,
            horizon=prediction_length,
            stride=stride,
        )
        preds = forecast_batches(
            pipeline=pipeline,
            contexts=contexts,
            horizon=prediction_length,
            batch_size=batch_size,
            num_samples=num_samples,
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


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def load_pipeline(
    model_id: str,
    pipeline_kind: str,
    device_map: str,
    torch_dtype: str,
) -> PipelineType:
    if pipeline_kind == "auto":
        use_chronos2 = "chronos-2" in model_id.lower() or "chronos2" in model_id.lower()
        pipeline_cls = Chronos2Pipeline if use_chronos2 else ChronosPipeline
    elif pipeline_kind == "chronos":
        pipeline_cls = ChronosPipeline
    elif pipeline_kind == "chronos2":
        pipeline_cls = Chronos2Pipeline
    else:
        raise ValueError(f"Unknown pipeline kind: {pipeline_kind}")

    kwargs = {}
    if device_map:
        kwargs["device_map"] = device_map
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = resolve_torch_dtype(torch_dtype)
    return pipeline_cls.from_pretrained(model_id, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chronos zero-shot evaluation on ETT datasets."
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--pipeline",
        type=str,
        default="auto",
        choices=["auto", "chronos", "chronos2"],
        help="Pipeline implementation to use.",
    )
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
        "--num-samples",
        type=int,
        default=20,
        help="Number of sample paths for ChronosPipeline (ignored by Chronos2Pipeline).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prediction_lengths = parse_int_list(args.prediction_lengths)
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for dataset_name in dataset_names:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    print("Loading model...")
    pipeline = load_pipeline(
        model_id=args.model_id,
        pipeline_kind=args.pipeline,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    print("Loading datasets...")
    dataset_cache: Dict[str, Tuple[Dict[str, np.ndarray], List[str]]] = {}
    for dataset_name in dataset_names:
        dataset_cache[dataset_name] = load_dataset(DATASETS[dataset_name])

    all_results: List[EvalResult] = []
    for dataset_name in dataset_names:
        series_map, target_columns = dataset_cache[dataset_name]
        for pred_len in prediction_lengths:
            stride = args.stride if args.stride > 0 else pred_len
            print(
                f"\nRunning {dataset_name} with context={args.context_length}, "
                f"prediction={pred_len}, stride={stride}"
            )
            result = evaluate_dataset(
                pipeline=pipeline,
                dataset_name=dataset_name,
                series_map=series_map,
                target_columns=target_columns,
                context_length=args.context_length,
                prediction_length=pred_len,
                batch_size=args.batch_size,
                stride=stride,
                num_samples=args.num_samples,
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
