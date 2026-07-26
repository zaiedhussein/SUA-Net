from __future__ import annotations

import pytest

from suanet.cli import ablation, audit, failure_cases, generalization, suite, train


@pytest.mark.parametrize(
    "module",
    [ablation, audit, failure_cases, generalization, suite, train],
)
def test_cli_help(module, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [module.__name__, "--help"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out.lower()
