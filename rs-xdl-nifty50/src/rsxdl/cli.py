"""Command line entry for the Nifty 50 RS-XDL run."""

from __future__ import annotations

import argparse
import json

from rsxdl.pipeline import RunConfig, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regime-Switching Explainable Deep Learning on the Nifty 50")
    parser.add_argument("--start", default="2015-01-01", help="First session to download (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Last session, exclusive, as Yahoo Finance expects")
    parser.add_argument("--tickers", default=None, help="Comma-separated NSE symbols without the .NS suffix")
    parser.add_argument("--output", default="artifacts", help="Directory for metrics, SHAP, and forecasts")
    parser.add_argument("--quick", action="store_true", help="Eight large-cap names from 2023, fewer trees and epochs")
    parser.add_argument("--no-lstm", action="store_true", help="Fit only the regime-switching tree models")
    parser.add_argument("--epochs", type=int, default=None, help="LSTM training epochs")
    parser.add_argument("--macro", action="store_true", help="Append macro/VIX features (Yahoo Finance)")
    parser.add_argument("--walk-forward", action="store_true", help="5-fold expanding walk-forward; writes walk_forward_*.csv/json")
    parser.add_argument("--walk-forward-folds", type=int, default=5, help="Number of walk-forward test windows")
    parser.add_argument(
        "--walk-forward-only",
        action="store_true",
        help="Skip hold-out SHAP/portfolio after walk-forward",
    )
    parser.add_argument("--figures", action="store_true", help="Write PNGs under artifacts/figures/")
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Walk-forward + hold-out artifacts + figures (PhD default)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = RunConfig(
        start=args.start,
        end=args.end,
        output_dir=args.output,
        lstm=not args.no_lstm,
        use_macro=args.macro,
        walk_forward=args.walk_forward,
        walk_forward_folds=args.walk_forward_folds,
        walk_forward_holdout=not args.walk_forward_only,
        figures=args.figures,
    )
    if args.complete:
        config.complete()
        if args.macro:
            config.use_macro = True
        config.lstm = not args.no_lstm
    if args.tickers:
        config.symbols = [part.strip().upper() for part in args.tickers.split(",") if part.strip()]
    if args.quick:
        config.quick()
        if args.tickers:
            config.symbols = [part.strip().upper() for part in args.tickers.split(",") if part.strip()]
        if args.start != "2015-01-01":
            config.start = args.start
    if args.epochs is not None:
        config.lstm_epochs = args.epochs
    if args.no_lstm:
        config.lstm = False
    metrics = run(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
