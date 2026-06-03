"""CLI entrypoint:  dscribe run <patient.pdf>  [--no-cache] [--out summary.md]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CONFIG
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dscribe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Draft a discharge summary from a PDF bundle.")
    run_p.add_argument("pdf", help="Path to the patient's source-note PDF bundle.")
    run_p.add_argument("--no-cache", action="store_true",
                       help="Re-extract pages instead of using the cached OCR.")
    run_p.add_argument("--out", help="Write the Markdown draft to this path.")

    args = parser.parse_args(argv)

    if not CONFIG.openai_api_key:
        print("ERROR: OPENAI_API_KEY is not set (see .env.example).", file=sys.stderr)
        return 2

    if args.command == "run":
        if not Path(args.pdf).exists():
            print(f"ERROR: file not found: {args.pdf}", file=sys.stderr)
            return 2
        result = run(args.pdf, use_cache=not args.no_cache, echo_trace=True)
        print("\n" + "=" * 70)
        print(result.markdown)
        out = args.out or str(CONFIG.storage_dir / (Path(args.pdf).stem + ".md"))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(result.markdown)
        Path(out).with_suffix(".json").write_text(
            result.summary.model_dump_json(indent=2)
        )
        print(f"\nSaved draft -> {out}")
        print(f"Trace steps: {len(result.trace)}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
