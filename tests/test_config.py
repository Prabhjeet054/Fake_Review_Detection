"""Tests for src.config."""

from src.config import dump_config, get_nested, label_names, load_config, path_from_cfg, project_root


def test_load_config_attaches_root(cfg: dict) -> None:
    """Loaded config includes the project root."""
    assert "_project_root" in cfg
    assert project_root(cfg).exists()


def test_get_nested_and_paths(cfg: dict) -> None:
    """Dotted lookup and path resolution work."""
    assert get_nested(cfg, "project.seed") == 42
    assert get_nested(cfg, "missing.key", "fallback") == "fallback"
    figures = path_from_cfg(cfg, "paths.figures")
    assert figures.name == "figures"


def test_label_names(cfg: dict) -> None:
    """Label ids map to the three configured display names."""
    names = label_names(cfg)
    assert set(names) == {0, 1, 2}
    assert "Genuine" in names[0]


def test_dump_config_hides_internal(cfg: dict) -> None:
    """Serialized config does not include internal keys."""
    dumped = dump_config(cfg)
    assert "_project_root" not in dumped


def test_main_prints_seed(config_path: str, capsys) -> None:
    """CLI prints the resolved project root."""
    from src.config import main

    main(["--config", config_path])
    captured = capsys.readouterr().out
    assert "project_root=" in captured
