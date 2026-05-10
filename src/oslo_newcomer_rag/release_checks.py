from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


PLACEHOLDER_VALUES = {
    "",
    "change-me",
    "changeme",
    "example",
    "password",
    "pass",
    "replace-with-api-key",
    "replace-with-password",
    "test",
    "test-key",
}

ENV_PATH_PATTERNS = (
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)\.env\.(?!example$)[^/]+$"),
)

SECRET_PATTERNS = (
    ("private key block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("OpenAI-style API key", re.compile(r"\b" + "sk-" + r"[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"\b" + "gh[pousr]_" + r"[A-Za-z0-9_]{20,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("generic secret assignment", re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\s*=\s*['\"]([^'\"]{12,})['\"]")),
)

PERSONAL_NUMBER_PATTERN = re.compile(r"\b\d{6}[ -]?\d{5}\b")

TEXT_RISK_PATTERNS = (
    ("unsafe yaml load", re.compile(r"\byaml\.load\s*\(")),
    ("pickle deserialization", re.compile(r"\bpickle\.loads\s*\(")),
    ("dynamic eval", re.compile(r"\beval\s*\(")),
    ("dynamic exec", re.compile(r"\bexec\s*\(")),
    ("raw HTML injection", re.compile(r"dangerously" + r"SetInnerHTML")),
    ("wildcard CORS origin", re.compile(r"allow_origins\s*=\s*\[\s*['\"]\*['\"]\s*\]")),
)

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".dockerfile",
    ".env",
    ".example",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    detail: str
    line: int | None = None

    def format(self) -> str:
        location = self.path if self.line is None else f"{self.path}:{self.line}"
        return f"{location} - {self.check}: {self.detail}"


def main() -> None:
    root = Path.cwd()
    findings = run_checks(root)

    if findings:
        print("Release check failed:")
        for finding in findings:
            print(f"  - {finding.format()}")
        raise SystemExit(1)

    print("Release check passed: no tracked secrets, unsafe settings, or private-data fixtures found.")


def run_checks(root: Path) -> list[Finding]:
    tracked_paths = _git_lines(root, "ls-files", "-z")
    workspace_only_paths = _local_exclude_patterns(root)
    findings: list[Finding] = []

    findings.extend(_check_tracked_paths(tracked_paths, workspace_only_paths))
    findings.extend(_check_gitignore(root, workspace_only_paths))
    findings.extend(_scan_tracked_files(root, tracked_paths))
    findings.extend(_scan_staged_added_lines(root))

    return findings


def _git_lines(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )
    raw = result.stdout
    if args and args[-1] == "-z":
        return [part.decode("utf-8") for part in raw.split(b"\0") if part]
    return [line.decode("utf-8") for line in raw.splitlines()]


def _check_tracked_paths(paths: list[str], workspace_only_paths: set[str] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    normalized = {_normalize_path(path) for path in paths}
    workspace_paths = workspace_only_paths or set()

    for path in sorted(normalized):
        if _is_env_file(path):
            findings.append(Finding("tracked environment file", path, "environment files must stay local"))
        if _matches_workspace_only_path(path, workspace_paths):
            findings.append(Finding("tracked local planning file", path, "keep this out of repository history"))

    return findings


def _check_gitignore(root: Path, workspace_only_paths: set[str] | None = None) -> list[Finding]:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []

    workspace_paths = workspace_only_paths or set()
    findings: list[Finding] = []
    for line_number, raw_line in enumerate(gitignore.read_text(encoding="utf-8").splitlines(), start=1):
        pattern = raw_line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        if _matches_workspace_only_path(pattern, workspace_paths):
            findings.append(
                Finding(
                    "local planning ignore rule",
                    ".gitignore",
                    "put workspace-only planning files in .git/info/exclude instead",
                    line_number,
                )
            )

    return findings


def _local_exclude_patterns(root: Path) -> set[str]:
    exclude_file = root / ".git" / "info" / "exclude"
    if not exclude_file.exists():
        return set()

    patterns: set[str] = set()
    for raw_line in exclude_file.read_text(encoding="utf-8").splitlines():
        pattern = raw_line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        patterns.add(_normalize_path(pattern))

    return patterns


def _scan_tracked_files(root: Path, paths: list[str]) -> list[Finding]:
    findings: list[Finding] = []

    for path in paths:
        normalized = _normalize_path(path)
        file_path = root / normalized
        if not _looks_like_text(file_path):
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        findings.extend(_scan_text(normalized, text))

    return findings


def _scan_staged_added_lines(root: Path) -> list[Finding]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--unified=0", "--no-ext-diff"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    added_lines = [
        line[1:]
        for line in result.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    if not added_lines:
        return []

    return _scan_text("staged diff", "\n".join(added_lines))


def _scan_text(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if _is_comment_or_placeholder(line):
            continue

        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line) and not _is_allowed_secret_line(line):
                findings.append(Finding("secret-looking value", path, label, line_number))

        if PERSONAL_NUMBER_PATTERN.search(line):
            findings.append(
                Finding("personal-number-shaped text", path, "use a placeholder instead", line_number)
            )

        if _has_real_password_in_database_url(line):
            findings.append(
                Finding("hardcoded database password", path, "use environment variables or placeholders", line_number)
            )

        for label, pattern in TEXT_RISK_PATTERNS:
            if pattern.search(line):
                findings.append(Finding("unsafe code or config pattern", path, label, line_number))

    return findings


def _is_comment_or_placeholder(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#") and "example" in stripped.lower():
        return True
    return False


def _is_allowed_secret_line(line: str) -> bool:
    lowered = line.lower()
    allowed_fragments = (
        "replace-with",
        "change-me",
        "test-key",
        "provider.example",
        "api.example.com",
        "example.com",
        "user:pass@",
        "${",
    )
    return any(fragment in lowered for fragment in allowed_fragments)


def _has_real_password_in_database_url(line: str) -> bool:
    matches = re.finditer(r"postgres(?:ql)?(?:\+\w+)?://[^\s'\"<>]+", line)
    for match in matches:
        parsed = urlparse(match.group(0))
        if not parsed.password:
            continue
        if parsed.password.lower() not in PLACEHOLDER_VALUES:
            return True
    return False


def _looks_like_text(path: Path) -> bool:
    name = path.name.lower()
    if name in {"dockerfile", ".gitignore"}:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def _is_env_file(path: str) -> bool:
    if path == ".env.example" or path.endswith("/.env.example"):
        return False
    return any(pattern.search(path) for pattern in ENV_PATH_PATTERNS)


def _matches_workspace_only_path(path: str, workspace_only_paths: set[str]) -> bool:
    normalized = _normalize_path(path).rstrip("/")
    candidates = {item.rstrip("/") for item in workspace_only_paths}
    return any(normalized == item or normalized.startswith(f"{item}/") for item in candidates)


def _normalize_path(path: str) -> str:
    cleaned = path.strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


if __name__ == "__main__":
    sys.exit(main())
