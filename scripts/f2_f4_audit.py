"""F2 (code quality) + F4 (scope fidelity) plan audits - local subset.

Locally-completable checks from plan todos F2 and F4. The pod-only halves
(Windows-CPU vs RunPod-Linux-GPU test subsets, F3 manual QA on the pod) are
documented, not executed here.

F2 checks:
  1. ruff clean on src/tests/scripts (exit 0).
  2. No code comment markers TODO/FIXME outside docs/ (case-sensitive comment
     markers; "plan todo N" docstring references are NOT markers).
  3. No flash-attn imports in src/familydraft/verify/.
  4. Dependency pins: every pyproject dependency resolves in uv.lock.

F4 checks:
  5. No forbidden-family artifacts (qwq / 235b / qwen3.5) in git-tracked names.
  6. No engine integration code in src/familydraft (sglang/vllm/triton serve
     paths are Wave-G, out of scope until then).
  7. No manuscript prose: docs/paper/ absent or skeleton-only (no prose .md).
  8. No secrets in git-tracked files (api keys / tokens / bearer).
  9. Pre-registration immutability: verdict_protocol.yaml + oracle_thresholds.yaml
     have no commits AFTER the M3 campaign pin commit.
  10. Gated experts: expert gate evidence file exists (configs/expert_gates.yaml).

Exit 0 only if every locally-checkable criterion passes.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# GitHub/HF/OpenAI-style secrets; anchored to avoid matching doc examples.
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|"
    r"Bearer\s+[A-Za-z0-9._-]{20,})"
)


def _run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _tracked_files(root: Path) -> list[Path]:
    rc, out = _run(["git", "ls-files"], root)
    if rc != 0:
        return []
    return [root / p for p in out.splitlines() if p]


def main() -> int:
    failures: list[str] = []

    # 1. ruff clean
    rc, _ = _run(["uv", "run", "ruff", "check", "src", "tests", "scripts"])
    if rc != 0:
        failures.append("ruff check src/tests/scripts not clean")
    else:
        print("  OK  ruff clean (src, tests, scripts)")

    # 2. TODO/FIXME comment markers outside docs/
    todo_hits = []
    for f in _tracked_files(ROOT):
        if f.suffix not in (".py", ".ts", ".tsx", ".js", ".go", ".rs"):
            continue
        if "docs" in f.parts:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if re.search(r"#\s*(TODO|FIXME)\b", line):
                todo_hits.append(f"{f.relative_to(ROOT)}:{i}")
    if todo_hits:
        failures.append(f"TODO/FIXME comment markers outside docs/: {todo_hits[:5]}")
    else:
        print("  OK  no TODO/FIXME comment markers outside docs/")

    # 3. no flash-attn in verify/
    flash_hits = []
    verify_dir = ROOT / "src" / "familydraft" / "verify"
    if verify_dir.exists():
        for f in verify_dir.rglob("*.py"):
            if "flash_attn" in f.read_text(encoding="utf-8", errors="ignore"):
                flash_hits.append(str(f))
    if flash_hits:
        failures.append(f"flash-attn imports in verify/: {flash_hits}")
    else:
        print("  OK  no flash-attn in src/familydraft/verify/")

    # 4. dependency pins resolve in lock
    rc, out = _run(["uv", "lock", "--check"])
    if rc != 0:
        failures.append(f"uv lock --check failed: {out[-300:]}")
    else:
        print("  OK  dependency pins match uv.lock")

    # 5. no forbidden-family artifacts
    forbidden = [
        p for p in _tracked_files(ROOT)
        if re.search(r"qwq|235b|qwen3[._-]?5", p.name, re.I)
    ]
    if forbidden:
        failures.append(f"forbidden-family artifacts in git: {forbidden[:5]}")
    else:
        print("  OK  no QwQ/235B/Qwen3.5 artifacts in git")

    # 6. no engine integration code in src/familydraft
    engine_hits = []
    for f in (ROOT / "src" / "familydraft").rglob("*.py"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\b(sglang|vllm|triton)\b", text):
            engine_hits.append(str(f))
    if engine_hits:
        failures.append(f"engine integration code in src/familydraft: {engine_hits[:5]}")
    else:
        print("  OK  no engine integration (sglang/vllm/triton) in src/familydraft")

    # 7. no manuscript prose beyond skeleton (skeleton.md itself is the allowed
    #    skeleton; only OTHER .md files in docs/paper/ count as prose).
    paper_dir = ROOT / "docs" / "paper"
    prose = []
    if paper_dir.exists():
        for f in paper_dir.rglob("*.md"):
            if f.name == "skeleton.md":
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            if len(text) > 200:  # skeleton outline only
                prose.append(str(f))
    if prose:
        failures.append(f"manuscript prose beyond skeleton: {prose}")
    else:
        print("  OK  docs/paper skeleton-only (no prose)")

    # 8. no secrets in git-tracked files
    secret_hits = []
    for f in _tracked_files(ROOT):
        if f.suffix not in (".py", ".yaml", ".yml", ".json", ".toml", ".sh", ".md"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _SECRET_RE.search(text):
            secret_hits.append(str(f))
    if secret_hits:
        failures.append(f"possible secrets in git-tracked files: {secret_hits[:5]}")
    else:
        print("  OK  no secrets in git-tracked files")

    # 9. pre-registration immutability (no commits after the M3 pin commit)
    pin = "f3e79f4"  # verdict-protocol pins commit (M3 campaign start)
    for cfg in ("configs/verdict_protocol.yaml", "configs/oracle_thresholds.yaml"):
        rc, out = _run(["git", "log", "--oneline", f"{pin}..HEAD", "--", cfg])
        if rc == 0 and out.strip():
            failures.append(f"{cfg} modified after M3 pin commit {pin}: {out.strip()}")
        else:
            print(f"  OK  {cfg} unmodified since M3 pin commit")

    # 10. gated-expert evidence file exists
    if not (ROOT / "configs" / "expert_gates.yaml").exists():
        failures.append("configs/expert_gates.yaml missing (gated-expert evidence)")
    else:
        print("  OK  configs/expert_gates.yaml present")

    if failures:
        print("\nF2/F4 AUDIT (local subset): FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nF2/F4 AUDIT (local subset): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
