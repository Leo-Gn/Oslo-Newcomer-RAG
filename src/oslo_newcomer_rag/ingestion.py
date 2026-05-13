from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup, Tag
from sqlalchemy import select
from sqlalchemy.orm import Session

from oslo_newcomer_rag.config import Settings, get_settings
from oslo_newcomer_rag.db.models import Document, DocumentChunk, Source
from oslo_newcomer_rag.db.session import create_engine_from_settings
from oslo_newcomer_rag.sources import OFFICIAL_DOMAINS, SourceEntry, load_source_registry


DEFAULT_USER_AGENT = "OsloNewcomerRAG/0.1 static snapshot (+https://github.com/)"
MAX_SECTION_WORDS = 450
SECTION_OVERLAP_WORDS = 60
MAX_REDIRECTS = 5
MAX_SOURCE_BYTES = 2_000_000


@dataclass(frozen=True)
class FetchedPage:
    url: str
    html: str
    last_modified_header: str | None = None


@dataclass(frozen=True)
class ParsedSection:
    heading: str
    url: str
    text: str


@dataclass(frozen=True)
class ParsedPage:
    title: str | None
    canonical_url: str
    language: str
    official_last_updated_at: datetime | None
    sections: list[ParsedSection]
    raw_text: str
    content_hash: str


@dataclass(frozen=True)
class IngestedSourceResult:
    source_id: str
    url: str
    status: str
    document_id: str | None
    chunks_written: int
    content_hash: str | None


@dataclass(frozen=True)
class IngestionRunResult:
    collected_at: datetime
    fetched: int
    inserted_documents: int
    skipped_documents: int
    chunks_written: int
    results: list[IngestedSourceResult]


PageFetcher = Callable[[SourceEntry], FetchedPage]


class SnapshotFetchError(RuntimeError):
    pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def fetch_source_page(source: SourceEntry, timeout: float = 25.0) -> FetchedPage:
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    allowed_hosts = OFFICIAL_DOMAINS[source.owner]
    current_url = source.url
    with httpx.Client(follow_redirects=False, timeout=timeout, headers=headers) as client:
        for redirect_count in range(MAX_REDIRECTS + 1):
            _validate_official_https_url(current_url, allowed_hosts, source.id)
            response = client.get(current_url)
            if not response.is_redirect:
                break
            if redirect_count == MAX_REDIRECTS:
                raise SnapshotFetchError(f"{source.id} redirected too many times")
            location = response.headers.get("location")
            if not location:
                raise SnapshotFetchError(f"{source.id} returned a redirect without a location")
            current_url = urljoin(str(response.url), location)
        else:
            raise SnapshotFetchError(f"{source.id} redirected too many times")

        response.raise_for_status()

    final_url = str(response.url)
    _validate_official_https_url(final_url, allowed_hosts, source.id)
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_SOURCE_BYTES:
                raise SnapshotFetchError(f"{source.id} response was larger than the snapshot limit")
        except ValueError:
            raise SnapshotFetchError(f"{source.id} returned an invalid content-length header") from None
    if len(response.content) > MAX_SOURCE_BYTES:
        raise SnapshotFetchError(f"{source.id} response was larger than the snapshot limit")
    return FetchedPage(
        url=final_url,
        html=response.text,
        last_modified_header=response.headers.get("last-modified"),
    )


def parse_official_page(
    fetched: FetchedPage,
    language: str,
    allowed_hosts: frozenset[str] | None = None,
) -> ParsedPage:
    soup = BeautifulSoup(fetched.html, "html.parser")
    for unwanted in soup(["script", "style", "noscript", "svg", "iframe"]):
        unwanted.decompose()

    host_allowlist = allowed_hosts or _hosts_from_url(fetched.url)
    canonical_url = _canonical_url(soup, fetched.url, host_allowlist)
    official_last_updated_at = _detect_last_updated(soup, fetched.last_modified_header)
    title = _title(soup)
    content_root = soup.find("main") or soup.body or soup
    sections = _extract_sections(content_root, canonical_url, title)
    raw_text = "\n\n".join(section.text for section in sections)

    return ParsedPage(
        title=title,
        canonical_url=canonical_url,
        language=language,
        official_last_updated_at=official_last_updated_at,
        sections=sections,
        raw_text=raw_text,
        content_hash=sha256_text(raw_text),
    )


def chunk_section(section: ParsedSection) -> Iterable[ParsedSection]:
    words = section.text.split()
    if len(words) <= MAX_SECTION_WORDS:
        yield section
        return

    start = 0
    part = 1
    while start < len(words):
        end = min(start + MAX_SECTION_WORDS, len(words))
        heading = f"{section.heading} (part {part})"
        yield ParsedSection(
            heading=heading,
            url=section.url,
            text=" ".join(words[start:end]),
        )
        if end == len(words):
            break
        start = max(0, end - SECTION_OVERLAP_WORDS)
        part += 1


def ingest_registry(
    session: Session,
    fetcher: PageFetcher = fetch_source_page,
    collected_at: datetime | None = None,
    limit: int | None = None,
) -> IngestionRunResult:
    registry = load_source_registry()
    run_collected_at = collected_at or datetime.now(UTC)
    results: list[IngestedSourceResult] = []
    fetched_count = 0
    inserted_documents = 0
    skipped_documents = 0
    chunks_written = 0

    entries = registry.sources[:limit] if limit else registry.sources
    for entry in entries:
        source = _upsert_source(session, entry)
        fetched = fetcher(entry)
        parsed = parse_official_page(fetched, entry.language, allowed_hosts=OFFICIAL_DOMAINS[entry.owner])
        fetched_count += 1
        if not parsed.sections:
            results.append(
                IngestedSourceResult(
                    source_id=entry.id,
                    url=entry.url,
                    status="no_content",
                    document_id=None,
                    chunks_written=0,
                    content_hash=parsed.content_hash,
                )
            )
            continue

        existing_document = session.scalar(
            select(Document).where(
                Document.source_id == source.id,
                Document.content_hash == parsed.content_hash,
            )
        )
        if existing_document:
            skipped_documents += 1
            results.append(
                IngestedSourceResult(
                    source_id=entry.id,
                    url=entry.url,
                    status="skipped_existing",
                    document_id=str(existing_document.id),
                    chunks_written=0,
                    content_hash=parsed.content_hash,
                )
            )
            continue

        document = Document(
            source_id=source.id,
            title=parsed.title,
            canonical_url=parsed.canonical_url,
            language=parsed.language,
            collected_at=run_collected_at,
            official_last_updated_at=parsed.official_last_updated_at,
            content_hash=parsed.content_hash,
            raw_text=parsed.raw_text,
        )
        session.add(document)
        session.flush()

        chunk_index = 0
        for section in parsed.sections:
            for chunk in chunk_section(section):
                session.add(
                    DocumentChunk(
                        document_id=document.id,
                        source_id=source.id,
                        chunk_index=chunk_index,
                        section_heading=chunk.heading,
                        section_url=chunk.url,
                        language=parsed.language,
                        text=chunk.text,
                        text_hash=sha256_text(chunk.text),
                        token_count=len(chunk.text.split()),
                        collected_at=run_collected_at,
                        official_last_updated_at=parsed.official_last_updated_at,
                    )
                )
                chunk_index += 1

        inserted_documents += 1
        chunks_written += chunk_index
        results.append(
            IngestedSourceResult(
                source_id=entry.id,
                url=entry.url,
                status="inserted",
                document_id=str(document.id),
                chunks_written=chunk_index,
                content_hash=parsed.content_hash,
            )
        )

    return IngestionRunResult(
        collected_at=run_collected_at,
        fetched=fetched_count,
        inserted_documents=inserted_documents,
        skipped_documents=skipped_documents,
        chunks_written=chunks_written,
        results=results,
    )


def run_ingestion(settings: Settings, limit: int | None = None) -> IngestionRunResult:
    engine = create_engine_from_settings(settings)
    try:
        with Session(engine) as session:
            result = ingest_registry(session=session, limit=limit)
            session.commit()
            return result
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the one-time official source snapshot.")
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N registry entries.")
    args = parser.parse_args()

    result = run_ingestion(get_settings(), limit=args.limit)
    print(
        json.dumps(
            {
                "collected_at": result.collected_at.isoformat(),
                "fetched": result.fetched,
                "inserted_documents": result.inserted_documents,
                "skipped_documents": result.skipped_documents,
                "chunks_written": result.chunks_written,
                "results": [source.__dict__ for source in result.results],
            },
            indent=2,
        )
    )


def _upsert_source(session: Session, entry: SourceEntry) -> Source:
    source = session.scalar(select(Source).where(Source.url == entry.url))
    if source is None:
        source = Source(
            owner=entry.owner,
            url=entry.url,
            language=entry.language,
            category=entry.category,
            intended_coverage=entry.intended_coverage.model_dump(),
        )
        session.add(source)
        session.flush()
        return source

    source.owner = entry.owner
    source.language = entry.language
    source.category = entry.category
    source.intended_coverage = entry.intended_coverage.model_dump()
    session.flush()
    return source


def _canonical_url(soup: BeautifulSoup, fetched_url: str, allowed_hosts: frozenset[str]) -> str:
    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    if isinstance(canonical, Tag):
        href = canonical.get("href")
        if isinstance(href, str) and href.strip():
            return _safe_official_url(href.strip(), fetched_url, allowed_hosts)
    return _safe_official_url(fetched_url, fetched_url, allowed_hosts)


def _title(soup: BeautifulSoup) -> str | None:
    h1 = soup.find("h1")
    if h1:
        value = normalize_text(h1.get_text(" ", strip=True))
        if value:
            return value

    if soup.title and soup.title.string:
        value = normalize_text(soup.title.string)
        if value:
            return value

    return None


def _extract_sections(root: Tag, canonical_url: str, title: str | None) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    heading = title or "Page"
    section_url = canonical_url
    paragraphs: list[str] = []
    previous_text = ""

    def flush() -> None:
        nonlocal paragraphs
        text = normalize_text(" ".join(paragraphs))
        paragraphs = []
        if len(text.split()) >= 8:
            sections.append(ParsedSection(heading=heading, url=section_url, text=text))

    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        if _is_inside_ignored_region(element):
            continue

        text = normalize_text(element.get_text(" ", strip=True))
        if not text or text == previous_text:
            continue
        previous_text = text

        if element.name in {"h1", "h2", "h3", "h4"}:
            flush()
            heading = text
            section_url = _section_url(canonical_url, element)
            continue

        paragraphs.append(text)

    flush()
    if sections:
        return sections

    fallback_text = normalize_text(root.get_text(" ", strip=True))
    if fallback_text:
        return [ParsedSection(heading=heading, url=canonical_url, text=fallback_text)]
    return []


def _section_url(canonical_url: str, heading: Tag) -> str:
    anchor = heading.get("id")
    if not anchor:
        anchor_tag = heading.find("a", id=True) or heading.find("a", attrs={"name": True})
        if isinstance(anchor_tag, Tag):
            anchor = anchor_tag.get("id") or anchor_tag.get("name")

    if isinstance(anchor, str) and anchor.strip():
        return f"{canonical_url}#{quote(anchor.strip(), safe='-._~')}"
    return canonical_url


def _safe_official_url(candidate_url: str, fetched_url: str, allowed_hosts: frozenset[str]) -> str:
    fallback = _strip_url_noise(fetched_url)
    joined = urljoin(fetched_url, candidate_url)
    try:
        _validate_official_https_url(joined, allowed_hosts, "canonical")
    except SnapshotFetchError:
        return fallback
    return _strip_url_noise(joined)


def _strip_url_noise(value: str) -> str:
    parsed = urlparse(value)
    return urlunparse(parsed._replace(params="", query="", fragment=""))


def _validate_official_https_url(url: str, allowed_hosts: frozenset[str], source_id: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in allowed_hosts:
        raise SnapshotFetchError(f"{source_id} uses a non-allowlisted URL: {url}")
    if parsed.username or parsed.password:
        raise SnapshotFetchError(f"{source_id} URL must not contain credentials")
    try:
        has_port = parsed.port is not None
    except ValueError as exc:
        raise SnapshotFetchError(f"{source_id} URL has an invalid port") from exc
    if has_port:
        raise SnapshotFetchError(f"{source_id} URL must not contain an explicit port")


def _hosts_from_url(url: str) -> frozenset[str]:
    host = (urlparse(url).hostname or "").lower()
    return frozenset({host}) if host else frozenset()


def _is_inside_ignored_region(element: Tag) -> bool:
    ignored = {"nav", "header", "footer", "aside", "form", "button"}
    return any(parent.name in ignored for parent in element.parents if isinstance(parent, Tag))


def _detect_last_updated(soup: BeautifulSoup, last_modified_header: str | None) -> datetime | None:
    for meta_name in ("article:modified_time", "og:updated_time", "last-modified", "dateModified"):
        tag = soup.find("meta", attrs={"property": meta_name}) or soup.find("meta", attrs={"name": meta_name})
        if isinstance(tag, Tag):
            content = tag.get("content")
            parsed = _parse_date_value(content if isinstance(content, str) else None)
            if parsed:
                return parsed

    for script in soup.find_all("script", type="application/ld+json"):
        parsed = _date_from_json_ld(script.string)
        if parsed:
            return parsed

    for time_tag in soup.find_all("time"):
        datetime_value = time_tag.get("datetime")
        parsed = _parse_date_value(datetime_value if isinstance(datetime_value, str) else None)
        if parsed:
            return parsed

    page_text = normalize_text(soup.get_text(" ", strip=True))
    patterns = [
        r"(?:last updated|updated|published|sist oppdatert|oppdatert)\s*:?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
        r"(?:last updated|updated|published|sist oppdatert|oppdatert)\s*:?\s*([0-9]{1,2}\.?\s+[A-Za-zÆØÅæøå]+\s+[0-9]{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, page_text, flags=re.IGNORECASE)
        if match:
            parsed = _parse_date_value(match.group(1))
            if parsed:
                return parsed

    return _parse_http_date(last_modified_header)


def _date_from_json_ld(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    queue = data if isinstance(data, list) else [data]
    while queue:
        item = queue.pop(0)
        if not isinstance(item, dict):
            continue
        for key in ("dateModified", "datePublished"):
            value = item.get(key)
            parsed = _parse_date_value(value if isinstance(value, str) else None)
            if parsed:
                return parsed
        graph = item.get("@graph")
        if isinstance(graph, list):
            queue.extend(graph)
    return None


def _parse_http_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_date_value(value: str | None) -> datetime | None:
    if not value:
        return None

    clean = normalize_text(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(clean)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        pass

    numeric = re.match(r"^([0-9]{1,2})[./-]([0-9]{1,2})[./-]([0-9]{2,4})$", clean)
    if numeric:
        day, month, year = (int(group) for group in numeric.groups())
        if year < 100:
            year += 2000
        try:
            return datetime(year, month, day, tzinfo=UTC)
        except ValueError:
            return None

    month_match = re.match(r"^([0-9]{1,2})\.?\s+([A-Za-zÆØÅæøå]+)\s+([0-9]{4})$", clean)
    if month_match:
        day_raw, month_raw, year_raw = month_match.groups()
        month = _month_number(month_raw)
        if month:
            try:
                return datetime(int(year_raw), month, int(day_raw), tzinfo=UTC)
            except ValueError:
                return None

    return None


def _month_number(value: str) -> int | None:
    months = {
        "january": 1,
        "jan": 1,
        "januar": 1,
        "february": 2,
        "feb": 2,
        "februar": 2,
        "march": 3,
        "mar": 3,
        "mars": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "mai": 5,
        "june": 6,
        "jun": 6,
        "juni": 6,
        "july": 7,
        "jul": 7,
        "juli": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "october": 10,
        "oct": 10,
        "oktober": 10,
        "okt": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
        "desember": 12,
        "des": 12,
    }
    return months.get(value.lower())
