#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_LOG_FILES = [
    "chronos.log",
    "timecma.no_epoch.log",
    "timemoe.log",
    "timerxl.log",
    "timesfm.log",
]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

AVERAGE_HEADER = "Averaged across prediction lengths (dataset, avg_mse, avg_mae)"
SUMMARY_HEADER = "Summary (dataset, prediction_length, overall_mse, overall_mae)"

AVG_LINE_RE = re.compile(r"^(ETTh1|ETTh2|ETTm1|ETTm2),([0-9eE+.-]+),([0-9eE+.-]+)$")
SUMMARY_LINE_RE = re.compile(
    r"^(ETTh1|ETTh2|ETTm1|ETTm2),([0-9]+),([0-9eE+.-]+),([0-9eE+.-]+)$"
)


def model_name_from_file(path: Path) -> str:
    name = path.name
    if name.endswith(".no_epoch.log"):
        return name[: -len(".no_epoch.log")]
    if name.endswith(".log"):
        return name[: -len(".log")]
    return path.stem


def parse_log(path: Path) -> Dict[str, Tuple[float, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()

    parsed_avg = parse_averaged_section(lines)
    if parsed_avg:
        return parsed_avg

    parsed_summary = parse_summary_section(lines)
    if parsed_summary:
        return parsed_summary

    raise ValueError(f"Could not parse averages from {path}")


def parse_averaged_section(lines: List[str]) -> Dict[str, Tuple[float, float]]:
    start = -1
    for i, line in enumerate(lines):
        if line.strip() == AVERAGE_HEADER:
            start = i + 1
            break
    if start == -1:
        return {}

    out: Dict[str, Tuple[float, float]] = {}
    for line in lines[start:]:
        text = line.strip()
        if not text:
            continue
        if text.startswith("Summary ("):
            continue
        m = AVG_LINE_RE.match(text)
        if not m:
            # Stop when leaving the averaged block.
            if out:
                break
            continue
        ds = m.group(1)
        out[ds] = (float(m.group(2)), float(m.group(3)))
    return out


def parse_summary_section(lines: List[str]) -> Dict[str, Tuple[float, float]]:
    start = -1
    for i, line in enumerate(lines):
        if line.strip() == SUMMARY_HEADER:
            start = i + 1
            break
    if start == -1:
        return {}

    buckets: Dict[str, List[Tuple[float, float]]] = {d: [] for d in DATASETS}
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
        mse = float(m.group(3))
        mae = float(m.group(4))
        buckets[ds].append((mse, mae))

    out: Dict[str, Tuple[float, float]] = {}
    for ds, vals in buckets.items():
        if not vals:
            continue
        mse_avg = sum(v[0] for v in vals) / len(vals)
        mae_avg = sum(v[1] for v in vals) / len(vals)
        out[ds] = (mse_avg, mae_avg)
    return out


def build_table(rows: List[Tuple[str, Dict[str, Tuple[float, float]]]]) -> str:
    headers = ["Model"]
    for ds in DATASETS:
        headers.extend([f"{ds} MSE", f"{ds} MAE"])
    headers.extend(["Avg MSE", "Avg MAE"])

    table_rows: List[List[str]] = []
    numeric_rows: List[List[float]] = []
    for model, metrics in rows:
        line = [model]
        numeric_line: List[float] = []
        mse_vals: List[float] = []
        mae_vals: List[float] = []
        for ds in DATASETS:
            if ds in metrics:
                mse, mae = metrics[ds]
                line.extend([f"{mse:.6f}", f"{mae:.6f}"])
                numeric_line.extend([mse, mae])
                mse_vals.append(mse)
                mae_vals.append(mae)
            else:
                line.extend(["-", "-"])
                numeric_line.extend([float("inf"), float("inf")])
        if mse_vals and mae_vals:
            avg_mse = sum(mse_vals) / len(mse_vals)
            avg_mae = sum(mae_vals) / len(mae_vals)
            line.extend(
                [
                    f"{avg_mse:.6f}",
                    f"{avg_mae:.6f}",
                ]
            )
            numeric_line.extend([avg_mse, avg_mae])
        else:
            line.extend(["-", "-"])
            numeric_line.extend([float("inf"), float("inf")])
        table_rows.append(line)
        numeric_rows.append(numeric_line)

    # Rank by Avg MSE where available.
    def sort_key(row: List[str]) -> float:
        val = row[-2]
        return float(val) if val != "-" else float("inf")

    # Keep numeric rows aligned with table rows after sorting.
    order = sorted(range(len(table_rows)), key=lambda i: sort_key(table_rows[i]))
    table_rows = [table_rows[i] for i in order]
    numeric_rows = [numeric_rows[i] for i in order]

    # Bold best (lowest) value in each metric column.
    # Column 0 is model name; metric columns are 1..end.
    metric_col_count = len(headers) - 1
    col_mins = []
    for c in range(metric_col_count):
        min_val = min(row[c] for row in numeric_rows)
        col_mins.append(min_val)
    for r_idx, row in enumerate(table_rows):
        for c in range(metric_col_count):
            if numeric_rows[r_idx][c] == col_mins[c] and row[c + 1] != "-":
                row[c + 1] = f"**{row[c + 1]}**"

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a benchmark table from model log files."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory that contains the log files.",
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

    rows: List[Tuple[str, Dict[str, Tuple[float, float]]]] = []
    for file_name in args.files:
        path = args.logs_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing log file: {path}")
        rows.append((model_name_from_file(path), parse_log(path)))

    table = build_table(rows)
    print(table)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")
        print(f"\nSaved table to: {args.output}")


if __name__ == "__main__":
    main()
