import numpy as np
import pandas as pd
import torch
import timesfm

torch.set_float32_matmul_precision("high")

# Dataset (same as granite-zeroshot.py)
DATASET_PATH = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
TIMESTAMP_COLUMN = "date"
TARGET_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
SPLIT_CONFIG = {
    "train": [0, 8640],
    "valid": [8640, 11520],
    "test": [11520, 14400],
}

# Forecast config
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96
STRIDE = PREDICTION_LENGTH
BATCH_SIZE = 32


def build_eval_windows(series_values, start, end, context_length, horizon, stride):
    contexts = []
    targets = []
    for t in range(start, end - horizon + 1, stride):
        contexts.append(series_values[t - context_length : t])
        targets.append(series_values[t : t + horizon])
    return contexts, targets


def forecast_batches(model, contexts, horizon, batch_size):
    preds = []
    for i in range(0, len(contexts), batch_size):
        batch_contexts = contexts[i : i + batch_size]
        point_forecast, _ = model.forecast(horizon=horizon, inputs=batch_contexts)
        preds.append(point_forecast)
    if not preds:
        return np.empty((0, horizon), dtype=np.float32)
    return np.vstack(preds)


def main():
    data = pd.read_csv(DATASET_PATH, parse_dates=[TIMESTAMP_COLUMN])

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_LENGTH,
            max_horizon=PREDICTION_LENGTH,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    test_start, test_end = SPLIT_CONFIG["test"]
    overall_targets = []
    overall_preds = []
    per_series_mse = {}

    for col in TARGET_COLUMNS:
        series_values = data[col].to_numpy(dtype=np.float32)
        contexts, targets = build_eval_windows(
            series_values,
            start=test_start,
            end=test_end,
            context_length=CONTEXT_LENGTH,
            horizon=PREDICTION_LENGTH,
            stride=STRIDE,
        )

        preds = forecast_batches(
            model, contexts, horizon=PREDICTION_LENGTH, batch_size=BATCH_SIZE
        )
        targets_np = np.stack(targets) if targets else np.empty((0, PREDICTION_LENGTH))

        if preds.shape != targets_np.shape:
            raise ValueError(
                f"Prediction/target shape mismatch for {col}: "
                f"{preds.shape} vs {targets_np.shape}"
            )

        mse = float(np.mean((preds - targets_np) ** 2)) if preds.size else float("nan")
        per_series_mse[col] = mse
        overall_preds.append(preds)
        overall_targets.append(targets_np)

    if overall_preds:
        overall_preds_np = np.vstack(overall_preds)
        overall_targets_np = np.vstack(overall_targets)
        overall_mse = float(np.mean((overall_preds_np - overall_targets_np) ** 2))
    else:
        overall_mse = float("nan")

    print("Per-series MSE (zero-shot, TimesFM):")
    print(per_series_mse.items())
    print(f"Overall MSE: {overall_mse:.6f}")


if __name__ == "__main__":
    main()
