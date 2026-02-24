#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_LOG_FILES = [
    "chronos.log",
    "timecma.no_epoch.log",
    "timemoe.log",
    "timerxl.log",
    "timesfm.log",
]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

SUMMARY_HEADER = "Summary (dataset, prediction_length, overall_mse, overall_mae)"
SUMMARY_LINE_RE = re.compile(
    r"^(ETTh1|ETTh2|ETTm1|ETTm2),([0-9]+),([0-9eE+.-]+),([0-9eE+.-]+)$"
)
RUN_LINE_RE = re.compile(
    r"^\[(ETTh1|ETTh2|ETTm1|ETTm2)\]\s+pred=([0-9]+).*overall_mse=([0-9eE+.-]+)\s+overall_mae=([0-9eE+.-]+)$"
)
CONTEXT_RE = re.compile(r"context=([0-9]+)")


PairKey = Tuple[str, int]  # (dataset, horizon)
PairMetric = Tuple[float, float]  # (mse, mae)


def model_name_from_file(path: Path) -> str:
    name = path.name
    if name.endswith(".no_epoch.log"):
        return name[: -len(".no_epoch.log")]
    if name.endswith(".log"):
        return name[: -len(".log")]
    return path.stem


def parse_log(path: Path) -> Tuple[Dict[PairKey, PairMetric], List[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    contexts = parse_context_lengths(lines)
    by_pair = parse_summary_section(lines)
    if by_pair:
        return by_pair, contexts
    by_pair = parse_run_lines(lines)
    if by_pair:
        return by_pair, contexts
    raise ValueError(f"Could not parse benchmark rows from {path}")


def parse_summary_section(lines: Sequence[str]) -> Dict[PairKey, PairMetric]:
    starts = [i + 1 for i, line in enumerate(lines) if line.strip() == SUMMARY_HEADER]
    if not starts:
        return {}
    # Use the latest summary block if logs contain multiple runs.
    start = starts[-1]
    out: Dict[PairKey, PairMetric] = {}
    for line in lines[start:]:
        text = line.strip()
        if not text:
            continue
        if text.startswith("Averaged across prediction lengths"):
            break
        m = SUMMARY_LINE_RE.match(text)
        if not m:
            continue
        ds = m.group(1)
        horizon = int(m.group(2))
        mse = float(m.group(3))
        mae = float(m.group(4))
        out[(ds, horizon)] = (mse, mae)
    return out


def parse_run_lines(lines: Sequence[str]) -> Dict[PairKey, PairMetric]:
    out: Dict[PairKey, PairMetric] = {}
    for line in lines:
        m = RUN_LINE_RE.match(line.strip())
        if not m:
            continue
        ds = m.group(1)
        horizon = int(m.group(2))
        mse = float(m.group(3))
        mae = float(m.group(4))
        out[(ds, horizon)] = (mse, mae)
    return out


def parse_context_lengths(lines: Sequence[str]) -> List[int]:
    values = set()
    for line in lines:
        m = CONTEXT_RE.search(line)
        if m:
            values.add(int(m.group(1)))
    return sorted(values)


def format_contexts(contexts: Sequence[int]) -> str:
    if not contexts:
        return "unknown"
    return ",".join(str(x) for x in contexts)


def collect_horizons(
    rows: Sequence[Tuple[str, Dict[PairKey, PairMetric], List[int]]]
) -> List[int]:
    horizons = set()
    for _model, metrics, _contexts in rows:
        for _ds, horizon in metrics:
            horizons.add(horizon)
    return sorted(horizons)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def build_history_section(rows: List[Tuple[str, Dict[PairKey, PairMetric], List[int]]]) -> str:
    lines = []
    lines.append("## History Length (Context)")
    lines.append("| Model | History Length |")
    lines.append("| --- | --- |")
    for model, _metrics, contexts in rows:
        lines.append(f"| {model} | {format_contexts(contexts)} |")
    return "\n".join(lines)


def build_tables_by_horizon(rows: List[Tuple[str, Dict[PairKey, PairMetric], List[int]]]) -> str:
    models = [model for model, _metrics, _contexts in rows]
    model_labels = [
        f"{model} (ctx={format_contexts(contexts)})"
        for model, _metrics, contexts in rows
    ]
    horizons = collect_horizons(rows)
    sections: List[str] = []
    for horizon in horizons:
        headers = ["Dataset"]
        for label in model_labels:
            headers.extend([f"{label} MSE", f"{label} MAE"])

        lines: List[str] = []
        lines.append(f"## Horizon {horizon}")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        best_notes: List[str] = []

        for dataset in DATASETS:
            row = [dataset]
            mse_vals: List[float] = []
            mae_vals: List[float] = []
            entries: List[Tuple[str, float, float]] = []
            for _model, metrics, _contexts in rows:
                metric = metrics.get((dataset, horizon))
                if metric is None:
                    row.extend(["-", "-"])
                    continue
                mse, mae = metric
                row.extend([format_float(mse), format_float(mae)])
                mse_vals.append(mse)
                mae_vals.append(mae)
                entries.append((_model, mse, mae))

            if mse_vals:
                best_mse = min(mse_vals)
                best_mae = min(mae_vals)
                best_mse_models = [
                    model_name for model_name, mse, _mae in entries if abs(mse - best_mse) < 1e-12
                ]
                best_mae_models = [
                    model_name for model_name, _mse, mae in entries if abs(mae - best_mae) < 1e-12
                ]
                for i in range(len(model_labels)):
                    mse_col = 1 + i * 2
                    mae_col = mse_col + 1
                    if row[mse_col] != "-" and abs(float(row[mse_col]) - best_mse) < 1e-12:
                        row[mse_col] = f"**{row[mse_col]}**"
                    if row[mae_col] != "-" and abs(float(row[mae_col]) - best_mae) < 1e-12:
                        row[mae_col] = f"**{row[mae_col]}**"
                best_notes.append(
                    f"- {dataset}: MSE -> {', '.join(best_mse_models)}; "
                    f"MAE -> {', '.join(best_mae_models)}"
                )
            lines.append("| " + " | ".join(row) + " |")

        if best_notes:
            lines.append("")
            lines.append("Best by dataset:")
            lines.extend(best_notes)

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build benchmark tables from model log files."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing log files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_LOG_FILES,
        help="Log filenames relative to --logs-dir.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output markdown file path.",
    )
    args = parser.parse_args()

    rows: List[Tuple[str, Dict[PairKey, PairMetric], List[int]]] = []
    for file_name in args.files:
        path = args.logs_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing log file: {path}")
        metrics, contexts = parse_log(path)
        rows.append((model_name_from_file(path), metrics, contexts))

    text = build_history_section(rows) + "\n\n" + build_tables_by_horizon(rows)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved table to: {args.output}")


if __name__ == "__main__":
    main()
