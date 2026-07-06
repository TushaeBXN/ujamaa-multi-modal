"""
Import AnthosWorlds bench traces as Ujamaa Phase-3 instruction data.

Usage:
    # readable instruction samples (what InstructionDataset loads)
    python scripts/import_worlds_traces.py \
        --traces ~/anthos-worlds/runs/oracle.jsonl \
        --output data/instruction/worlds.json

    # additionally pre-tokenize with response masks for instruction_collate
    python scripts/import_worlds_traces.py \
        --traces runs/oracle.jsonl --output data/instruction/worlds.json \
        --tokenized-output data/instruction/worlds_tokenized.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ujamaa.worlds_bridge import convert, load_traces, tokenize_episode


def main():
    parser = argparse.ArgumentParser(
        description="Convert anthos-worlds traces to Ujamaa instruction data.")
    parser.add_argument("--traces", required=True,
                        help="JSONL from anthos-worlds (raw traces or SFT export)")
    parser.add_argument("--output", required=True,
                        help="output JSON array for InstructionDataset")
    parser.add_argument("--tokenized-output", default=None,
                        help="also write input_ids + response_mask JSON here")
    parser.add_argument("--include-failures", action="store_true",
                        help="keep unsuccessful episodes (default: drop them)")
    parser.add_argument("--min-reward", type=float, default=None,
                        help="drop episodes below this reward")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    samples = convert(os.path.expanduser(args.traces),
                      successful_only=not args.include_failures,
                      min_reward=args.min_reward)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"wrote {len(samples)} instruction samples -> {args.output}")

    if args.tokenized_output:
        from ujamaa.utils.tokenizers import MultiModalTokenizer
        tokenizer = MultiModalTokenizer()
        kept = {s["task"] for s in samples}
        tokenized = [tokenize_episode(r, tokenizer, args.max_length)
                     for r in load_traces(os.path.expanduser(args.traces))
                     if r.get("task", "unknown") in kept or "messages" in r]
        os.makedirs(os.path.dirname(args.tokenized_output) or ".", exist_ok=True)
        with open(args.tokenized_output, "w", encoding="utf-8") as f:
            json.dump(tokenized, f)
        print(f"wrote {len(tokenized)} tokenized episodes -> {args.tokenized_output}")


if __name__ == "__main__":
    main()
