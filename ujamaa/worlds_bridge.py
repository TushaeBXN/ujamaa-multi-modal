"""
AnthosWorlds → Ujamaa Training Bridge

Converts agent episodes recorded by anthos-worlds (the AnthosWorlds bench at
~/anthos-worlds) into Ujamaa Phase-3 instruction samples, so bench runs feed
directly into instruction tuning: bench → traces → Ujamaa fine-tune → bench.

Accepts both trace formats anthos-worlds emits:
  - raw traces   {"env", "task", "instruction", "first_observation",
                  "steps": [{"action", "observation"}], "reward", "success"}
  - SFT exports  {"messages": [{"role", "content"}, ...]}

Episodes render as plain-text transcripts (GPT-2 BPE has no chat template):

    [WORLD] <environment description + available commands>
    [GOAL] <task instruction>
    [OBS] <observation>
    [ACT] <action>          <- response_mask = 1 only on these tokens
    ...

Tokenization is piecewise, so the response mask aligns exactly with the
action tokens Phase3Instruction trains on.

Usage:
    python scripts/import_worlds_traces.py --traces runs/oracle.jsonl \
        --output data/instruction/worlds.json
"""

import json
from typing import Dict, List, Optional, Tuple

Piece = Tuple[str, bool]  # (text, counts toward response loss)

_WORLD_INFO_FALLBACK = "A simulated text environment. Reply with one command per turn."


def _world_info(env_name: str) -> str:
    """Enrich transcripts with the environment's own description when the
    anthos_worlds package is importable; degrade gracefully when it isn't."""
    try:
        from anthos_worlds import make
        env = make(env_name)
        return f"{env.description} Commands: {env.actions_help()}"
    except Exception:
        return _WORLD_INFO_FALLBACK


def load_traces(path: str) -> List[Dict]:
    """Read an anthos-worlds JSONL file (raw traces or SFT export)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_messages(record: Dict) -> List[Dict[str, str]]:
    """Normalize either trace format to system/user/assistant messages."""
    if "messages" in record:
        return record["messages"]
    system = (f"{_world_info(record['env'])}\n"
              f"Your goal: {record['instruction']}")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": record["first_observation"]}]
    for step in record["steps"]:
        messages.append({"role": "assistant", "content": step["action"]})
        messages.append({"role": "user", "content": step["observation"]})
    # the trailing observation after the final action isn't a training target
    if messages[-1]["role"] == "user" and len(messages) > 2:
        messages.pop()
    return messages


def to_pieces(messages: List[Dict[str, str]]) -> List[Piece]:
    """Render messages as transcript pieces tagged response/not-response."""
    pieces: List[Piece] = []
    for m in messages:
        if m["role"] == "system":
            pieces.append((f"[WORLD] {m['content']}\n", False))
        elif m["role"] == "user":
            pieces.append((f"[OBS] {m['content']}\n", False))
        else:
            pieces.append((f"[ACT] {m['content']}\n", True))
    return pieces


def to_instruction_sample(record: Dict) -> Dict:
    """One episode -> one InstructionDataset sample (readable JSON form)."""
    messages = to_messages(record)
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    return {
        "source": "anthos-worlds",
        "env": record.get("env", "unknown"),
        "task": record.get("task", "unknown"),
        "reward": record.get("reward", 1.0),
        "instruction": system,
        "text": "".join(text for text, _ in to_pieces(messages)),
    }


def tokenize_episode(record: Dict, tokenizer, max_length: int = 2048) -> Dict:
    """One episode -> {"input_ids", "response_mask"} for instruction_collate.

    ``tokenizer`` is anything with ``encode(text) -> {"input_ids": [...]}``
    (MultiModalTokenizer works as is). Pieces are tokenized separately and
    concatenated so the mask is exact per token.
    """
    input_ids: List[int] = []
    response_mask: List[int] = []
    for text, is_response in to_pieces(to_messages(record)):
        ids = tokenizer.encode(text)["input_ids"]
        input_ids.extend(ids)
        response_mask.extend([1 if is_response else 0] * len(ids))
    return {"input_ids": input_ids[:max_length],
            "response_mask": response_mask[:max_length]}


def convert(traces_path: str, successful_only: bool = True,
            min_reward: Optional[float] = None) -> List[Dict]:
    """Load traces and return instruction samples ready to json.dump."""
    samples = []
    for record in load_traces(traces_path):
        reward = record.get("reward")
        if successful_only and record.get("success") is False:
            continue
        if min_reward is not None and reward is not None and reward < min_reward:
            continue
        samples.append(to_instruction_sample(record))
    return samples
