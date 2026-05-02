from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


OFFICIAL_DOMAINS: dict[str, frozenset[str]] = {
    "UDI": frozenset({"udi.no", "www.udi.no"}),
    "NAV": frozenset({"nav.no", "www.nav.no"}),
    "Skatteetaten": frozenset({"skatteetaten.no", "www.skatteetaten.no"}),
    "Oslo kommune": frozenset({"oslo.kommune.no", "www.oslo.kommune.no"}),
    "SUA": frozenset({"sua.no", "www.sua.no"}),
    "SiO": frozenset({"sio.no", "www.sio.no", "bolig.sio.no"}),
}

REQUIRED_OWNERS = frozenset(OFFICIAL_DOMAINS)

REQUIRED_COVERAGE = frozenset(
    {
        "permit_overview",
        "family_immigration",
        "work_immigration",
        "skilled_worker",
        "seasonal_worker",
        "study_permit",
        "job_seeker",
        "eu_eea_registration",
        "visitor_visa",
        "protection_asylum",
        "permanent_residence",
        "citizenship",
        "residence_card",
        "report_move_to_norway",
        "national_identity_number",
        "d_number",
        "tax_deduction_card",
        "national_insurance_scheme",
        "register_jobseeker",
        "service_centre_foreign_workers",
        "appointment_booking",
        "newcomer_basics",
        "first_steps",
        "healthcare_oslo",
        "housing_oslo",
        "children_families",
        "learning_norwegian",
        "student_welfare",
        "student_housing",
        "student_health",
    }
)


class SourceRegistryError(ValueError):
    pass


class IntendedCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    permit_paths: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

    @field_validator("permit_paths", "topics")
    @classmethod
    def values_must_be_tags(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value or value.strip() != value:
                raise ValueError("coverage tags must be non-empty and trimmed")
            if value.lower() != value or " " in value:
                raise ValueError("coverage tags must use lowercase snake_case")
        return values

    @model_validator(mode="after")
    def require_some_coverage(self) -> "IntendedCoverage":
        if not self.permit_paths and not self.topics:
            raise ValueError("intended_coverage must list at least one permit path or topic")
        return self

    @property
    def tags(self) -> set[str]:
        return set(self.permit_paths) | set(self.topics)


class SourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    owner: str
    url: str
    language: str
    category: str
    intended_coverage: IntendedCoverage

    @field_validator("id")
    @classmethod
    def id_must_be_slug(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("source id must be non-empty and trimmed")
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-")
        if any(char not in allowed for char in value):
            raise ValueError("source id must use lowercase kebab-case")
        return value

    @field_validator("owner")
    @classmethod
    def owner_must_be_allowed(cls, value: str) -> str:
        if value not in OFFICIAL_DOMAINS:
            raise ValueError(f"unsupported source owner: {value}")
        return value

    @field_validator("language")
    @classmethod
    def language_must_be_supported(cls, value: str) -> str:
        if value not in {"en", "nb", "nn"}:
            raise ValueError("language must be one of: en, nb, nn")
        return value

    @field_validator("category")
    @classmethod
    def category_must_be_snake_case(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("category must be non-empty and trimmed")
        if value.lower() != value or " " in value:
            raise ValueError("category must use lowercase snake_case")
        return value

    @model_validator(mode="after")
    def url_must_match_owner(self) -> "SourceEntry":
        parsed = urlparse(self.url)
        if parsed.scheme != "https":
            raise ValueError("source URLs must use https")
        if parsed.username or parsed.password:
            raise ValueError("source URLs must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("source URLs must not contain query strings or fragments")

        host = (parsed.hostname or "").lower()
        if host not in OFFICIAL_DOMAINS[self.owner]:
            raise ValueError(f"{self.owner} source uses non-official domain: {host}")

        return self

    @property
    def coverage_tags(self) -> set[str]:
        return self.intended_coverage.tags


class SourceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    description: str
    sources: list[SourceEntry]

    @model_validator(mode="after")
    def validate_registry(self) -> "SourceRegistry":
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source ids must be unique")

        urls = [source.url for source in self.sources]
        if len(urls) != len(set(urls)):
            raise ValueError("source URLs must be unique")

        owners = {source.owner for source in self.sources}
        missing_owners = REQUIRED_OWNERS - owners
        if missing_owners:
            missing = ", ".join(sorted(missing_owners))
            raise ValueError(f"registry is missing required owners: {missing}")

        coverage = self.coverage_tags
        missing_coverage = REQUIRED_COVERAGE - coverage
        if missing_coverage:
            missing = ", ".join(sorted(missing_coverage))
            raise ValueError(f"registry is missing required coverage: {missing}")

        return self

    @property
    def coverage_tags(self) -> set[str]:
        tags: set[str] = set()
        for source in self.sources:
            tags.update(source.coverage_tags)
        return tags


def default_sources_path() -> Path:
    return Path(__file__).resolve().parents[2] / "sources.yml"


def load_source_registry(path: Path | None = None) -> SourceRegistry:
    registry_path = path or default_sources_path()
    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SourceRegistryError(f"Could not read source registry: {registry_path}") from exc
    except yaml.YAMLError as exc:
        raise SourceRegistryError(f"Invalid YAML in source registry: {registry_path}") from exc

    if not isinstance(raw, dict):
        raise SourceRegistryError("Source registry must be a YAML mapping")

    try:
        return SourceRegistry.model_validate(raw)
    except ValueError as exc:
        raise SourceRegistryError(str(exc)) from exc


def registry_from_data(data: dict[str, Any]) -> SourceRegistry:
    try:
        return SourceRegistry.model_validate(data)
    except ValueError as exc:
        raise SourceRegistryError(str(exc)) from exc
