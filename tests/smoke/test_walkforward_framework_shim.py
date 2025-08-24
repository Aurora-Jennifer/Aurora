import warnings


def test_deprecation_shim_warns_and_forwards(monkeypatch, capsys):
    import scripts.walkforward_framework as shim

    captured = {}

    def fake_call(cmd, shell):  # noqa: ARG001
        captured["cmd"] = cmd
        return 0

    # Arrange
    monkeypatch.setattr(shim.subprocess, "call", fake_call)
    monkeypatch.setattr(shim.sys, "argv", ["scripts/walkforward_framework.py", "--smoke", "--validate-data"])  # type: ignore[attr-defined]

    # Act
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        try:
            shim.main()
        except SystemExit as e:  # expected from shim
            assert e.code == 0

    out = capsys.readouterr()

    # Assert: warning emitted
    assert any(isinstance(x.message, DeprecationWarning) or getattr(x, "category", None) is DeprecationWarning for x in w)
    assert "deprecated" in out.err.lower()

    # Assert: command forwarded with args
    assert "scripts/multi_walkforward_report.py" in captured.get("cmd", "")
    assert "--smoke" in captured.get("cmd", "")
    assert "--validate-data" in captured.get("cmd", "")


