"""Dedicated CLI for the single final residual-training trajectory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer_aer_residual_final import preflight_from_config, train_from_config
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=2)
    parser.add_argument("--preflight-output")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.preflight:
        if args.resume or args.stop_after_epoch or not args.preflight_output:
            parser.error("Preflight requires --preflight-output and forbids resume/stop-after-epoch")
        preflight_from_config(config, steps=args.preflight_steps, output_dir=args.preflight_output)
    else:
        result = train_from_config(config, resume=args.resume, stop_after_epoch=args.stop_after_epoch)
        print(f"Best trained residual checkpoint: {result if result is not None else 'no eligible trained candidate'}")


if __name__ == "__main__":
    main()
