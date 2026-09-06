"""Guards against a repository that works locally but is broken once cloned.

These exist because of a real failure: `.gitignore` contained the unanchored
pattern `data/`, which matched `mint_tts/data/` as well as the corpus
directory. The package imported fine in the working tree and every test
passed, but a fresh clone died with `ModuleNotFoundError: No module named
'mint_tts.data'` partway through preprocessing.
"""

import importlib
import pkgutil
import subprocess
from pathlib import Path

import pytest

import mint_tts

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "mint_tts"


def _tracked_files() -> set[str] | None:
    """Files git knows about, or None when this is not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if out.returncode != 0:
        return None
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def test_every_module_imports():
    failures = []
    for mod in pkgutil.walk_packages(mint_tts.__path__, "mint_tts."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # pragma: no cover - failure path
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)


def test_every_source_directory_is_a_package():
    missing = [
        str(d.relative_to(ROOT))
        for d in PACKAGE.rglob("*")
        if d.is_dir() and d.name != "__pycache__"
        and any(d.glob("*.py")) and not (d / "__init__.py").exists()
    ]
    assert not missing, f"directories with .py files but no __init__.py: {missing}"


def _package_sources() -> list[str]:
    return [
        p.relative_to(ROOT).as_posix()
        for p in PACKAGE.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def test_every_package_file_is_tracked_by_git():
    """A package file git does not track is a file a clone will not have."""
    tracked = _tracked_files()
    if tracked is None:
        pytest.skip("not a git checkout")
    untracked = [p for p in _package_sources() if p not in tracked]
    assert not untracked, (
        "these package files are missing from git, so a clone would not have "
        f"them: {untracked}\nCheck .gitignore for unanchored directory patterns."
    )


def test_no_package_file_matches_an_ignore_rule():
    """The invariant that actually matters.

    Tracking alone is not enough: once a file is tracked, `.gitignore` stops
    applying to it, so a bad rule can sit there unnoticed until someone
    re-adds the repo or exports it. This asks git directly whether any
    package source *would* be ignored.
    """
    sources = _package_sources()
    try:
        # bytes, not text: on Windows, text mode rewrites "\n" to "\r\n" on the
        # way in and git echoes the stray carriage returns straight back.
        out = subprocess.run(
            ["git", "check-ignore", "--stdin", "--no-index"],
            cwd=ROOT, input="\n".join(sources).encode(), capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pytest.skip("git unavailable")
    if out.returncode not in (0, 1):  # 0 = some ignored, 1 = none ignored
        pytest.skip("not a git checkout")
    ignored = [line.strip().strip('"') for line in out.stdout.decode().splitlines()
               if line.strip()]
    assert not ignored, (
        f"these package files match a .gitignore rule: {ignored}\n"
        "Anchor artefact patterns to the repo root (e.g. '/data/' not 'data/')."
    )


@pytest.mark.parametrize("script", sorted(p.name for p in (ROOT / "scripts").glob("*.py")))
def test_scripts_are_tracked_and_importable(script):
    tracked = _tracked_files()
    path = ROOT / "scripts" / script
    if tracked is not None:
        assert f"scripts/{script}" in tracked, f"scripts/{script} is not tracked by git"
    # --help exercises every module-level import without doing any work
    out = subprocess.run(
        ["python", str(path), "--help"], cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, f"{script} --help failed:\n{out.stderr[-800:]}"


@pytest.mark.parametrize("name", [
    "base.yaml", "exp0_dense.yaml", "exp2_token.yaml", "vctk_token.yaml",
    "libritts_token.yaml", "frontend_ipa.yaml",
])
def test_shipped_configs_are_tracked(name):
    tracked = _tracked_files()
    if tracked is None:
        pytest.skip("not a git checkout")
    assert f"configs/{name}" in tracked
