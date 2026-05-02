import pytest

from oslo_newcomer_rag.sources import (
    OFFICIAL_DOMAINS,
    REQUIRED_COVERAGE,
    REQUIRED_OWNERS,
    SourceRegistryError,
    load_source_registry,
    registry_from_data,
)


def test_default_source_registry_is_valid() -> None:
    registry = load_source_registry()

    assert registry.version == 1
    assert len(registry.sources) >= 20


def test_registry_contains_each_required_official_owner() -> None:
    registry = load_source_registry()
    owners = {source.owner for source in registry.sources}

    assert REQUIRED_OWNERS <= owners


def test_registry_urls_match_official_owner_domains() -> None:
    registry = load_source_registry()

    for source in registry.sources:
        assert source.owner in OFFICIAL_DOMAINS
        assert source.language
        assert source.category
        assert source.intended_coverage.tags


def test_registry_covers_main_permit_paths_and_newcomer_basics() -> None:
    registry = load_source_registry()

    assert REQUIRED_COVERAGE <= registry.coverage_tags


def test_registry_rejects_non_official_domains() -> None:
    bad_registry = {
        "version": 1,
        "description": "bad registry",
        "sources": [
            {
                "id": "bad-udi-source",
                "owner": "UDI",
                "url": "https://example.com/en/want-to-apply/",
                "language": "en",
                "category": "immigration",
                "intended_coverage": {
                    "permit_paths": ["permit_overview"],
                    "topics": ["newcomer_basics"],
                },
            }
        ],
    }

    with pytest.raises(SourceRegistryError, match="non-official domain"):
        registry_from_data(bad_registry)


def test_registry_rejects_missing_source_metadata() -> None:
    bad_registry = {
        "version": 1,
        "description": "bad registry",
        "sources": [
            {
                "id": "missing-category",
                "owner": "UDI",
                "url": "https://www.udi.no/en/want-to-apply/",
                "language": "en",
                "intended_coverage": {
                    "permit_paths": ["permit_overview"],
                    "topics": ["newcomer_basics"],
                },
            }
        ],
    }

    with pytest.raises(SourceRegistryError, match="category"):
        registry_from_data(bad_registry)
