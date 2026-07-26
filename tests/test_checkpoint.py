from pathlib import Path

import torch

from suanet.engine import load_checkpoint_state


def test_loads_legacy_and_structured_checkpoints(tmp_path: Path):
    state = {"weight": torch.ones(2)}
    raw_checkpoint = tmp_path / "raw_state_dict.pth"
    structured = tmp_path / "structured.pth"
    torch.save(state, raw_checkpoint)
    torch.save({"state_dict": state, "epoch": 3}, structured)

    raw_state, raw_meta = load_checkpoint_state(raw_checkpoint, torch.device("cpu"))
    structured_state, structured_meta = load_checkpoint_state(structured, torch.device("cpu"))
    assert torch.equal(raw_state["weight"], state["weight"])
    assert raw_meta["legacy_raw_state_dict"] is True
    assert structured_meta["epoch"] == 3
    assert torch.equal(structured_state["weight"], state["weight"])
