import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ujamaa.worlds_bridge import (convert, to_instruction_sample, to_messages,
                                  to_pieces, tokenize_episode)

RAW_TRACE = {
    "env": "quest", "task": "quest.rope",
    "instruction": "Retrieve the rope and bring it to the hall.",
    "reward": 1.0, "success": True,
    "first_observation": "You are in the hall.",
    "steps": [
        {"action": "go down", "observation": "You are in the cellar."},
        {"action": "take rope", "observation": "taken: rope"},
    ],
}

SFT_TRACE = {
    "messages": [
        {"role": "system", "content": "A test world.\nYour goal: win."},
        {"role": "user", "content": "obs one"},
        {"role": "assistant", "content": "act one"},
        {"role": "user", "content": "obs two"},
        {"role": "assistant", "content": "act two"},
    ],
}


class StubTokenizer:
    """Whitespace tokenizer with MultiModalTokenizer's encode interface."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text):
        ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]
        return {"input_ids": ids}


def test_raw_trace_to_messages_alternates_and_drops_trailing_obs():
    messages = to_messages(RAW_TRACE)
    assert messages[0]["role"] == "system"
    assert "Retrieve the rope" in messages[0]["content"]
    roles = [m["role"] for m in messages[1:]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert messages[-1]["content"] == "take rope"


def test_sft_trace_passes_through():
    assert to_messages(SFT_TRACE) == SFT_TRACE["messages"]


def test_instruction_sample_shape():
    sample = to_instruction_sample(RAW_TRACE)
    assert sample["source"] == "anthos-worlds"
    assert sample["env"] == "quest" and sample["reward"] == 1.0
    assert sample["text"].startswith("[WORLD] ")
    assert "[ACT] go down\n" in sample["text"]
    assert "[OBS] You are in the hall.\n" in sample["text"]


def test_response_mask_covers_exactly_the_action_tokens():
    tok = StubTokenizer()
    out = tokenize_episode(SFT_TRACE, tok)
    assert len(out["input_ids"]) == len(out["response_mask"])
    id_to_word = {v: k for k, v in tok.vocab.items()}
    masked = [id_to_word[i] for i, m in zip(out["input_ids"], out["response_mask"]) if m]
    assert masked == ["[ACT]", "act", "one", "[ACT]", "act", "two"]


def test_tokenize_respects_max_length():
    out = tokenize_episode(SFT_TRACE, StubTokenizer(), max_length=5)
    assert len(out["input_ids"]) == 5 and len(out["response_mask"]) == 5


def test_convert_filters_failures(tmp_path):
    failed = dict(RAW_TRACE, success=False, reward=0.0)
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps(RAW_TRACE) + "\n" + json.dumps(failed) + "\n")
    assert len(convert(str(traces))) == 1
    assert len(convert(str(traces), successful_only=False)) == 2
    assert len(convert(str(traces), successful_only=False, min_reward=0.5)) == 1


def test_pieces_tagging():
    pieces = to_pieces(SFT_TRACE["messages"])
    assert [flag for _, flag in pieces] == [False, False, True, False, True]
