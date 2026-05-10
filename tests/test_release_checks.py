from pathlib import Path

from oslo_newcomer_rag.release_checks import (
    _check_gitignore,
    _check_tracked_paths,
    _scan_text,
    run_checks,
)


def test_current_release_checks_pass() -> None:
    assert run_checks(Path.cwd()) == []


def test_tracked_environment_files_are_rejected() -> None:
    findings = _check_tracked_paths([".env", "frontend/.env.local", ".env.example"], set())

    assert [finding.path for finding in findings] == [".env", "frontend/.env.local"]


def test_local_workspace_files_are_rejected_from_gitignore(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text(".env\n.private-note\n.local-work/\n", encoding="utf-8")

    findings = _check_gitignore(tmp_path, {".private-note", ".local-work/"})

    assert len(findings) == 2
    assert {finding.line for finding in findings} == {2, 3}


def test_secret_like_values_are_reported_without_the_value() -> None:
    text = 'LLM_API_KEY = "sk-' + ("a" * 32) + '"'

    findings = _scan_text("settings.py", text)

    assert len(findings) == 1
    assert findings[0].check == "secret-looking value"
    assert "sk-" not in findings[0].format()


def test_personal_number_shape_is_reported() -> None:
    text = "Personal number: " + "120590" + " 12345"

    findings = _scan_text("case.txt", text)

    assert len(findings) == 1
    assert findings[0].check == "personal-number-shaped text"


def test_placeholder_database_urls_are_allowed() -> None:
    text = "\n".join(
        [
            "DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/oslo_newcomer",
            "DATABASE_URL=postgresql+psycopg://user:change-me@localhost:5432/oslo_newcomer",
        ]
    )

    assert _scan_text(".env.example", text) == []


def test_real_database_password_is_reported() -> None:
    database_url = (
        "DATABASE_URL=postgresql+psycopg://user:"
        + "real-password-123"
        + "@localhost:5432/app"
    )

    findings = _scan_text(
        "docker-compose.yml",
        database_url,
    )

    assert len(findings) == 1
    assert findings[0].check == "hardcoded database password"
