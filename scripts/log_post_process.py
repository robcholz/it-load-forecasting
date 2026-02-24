#!/usr/bin/env python3
import argparse
from pathlib import Path


def build_default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.no_epoch{input_path.suffix}")


def strip_epoch_lines(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if line.startswith("Epoch"):
                continue
            dst.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create a copy of a log file with lines starting with "Epoch" removed.'
    )
    parser.add_argument("input_file", type=Path, help="Path to the input log file.")
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <input_stem>.no_epoch<input_suffix>.",
    )
    args = parser.parse_args()

    input_path = args.input_file
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = args.output_file or build_default_output_path(input_path)
    strip_epoch_lines(input_path, output_path)
    print(f"Wrote filtered copy to: {output_path}")


if __name__ == "__main__":
    main()
