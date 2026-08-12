"""Convert GTIB long CSV tables into an Autogram-profiled pickle."""

from __future__ import annotations

import argparse
from pathlib import Path

from autogram.loader.gtib import prepare_gtib_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("derived", help="path to timeseries_derived.csv")
    parser.add_argument("--raw", default="", help="path to timeseries_raw.csv")
    parser.add_argument("--out", required=True, help="output pickle path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = prepare_gtib_files(args.derived, args.raw or None)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
