from datetime import UTC, datetime

from oslo_newcomer_rag.ingestion import FetchedPage, _parse_date_value, chunk_section, parse_official_page


SAMPLE_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <title>Working in Norway</title>
    <link rel="canonical" href="https://www.udi.no/en/want-to-apply/work-immigration/" />
    <meta property="article:modified_time" content="2026-01-15T09:30:00+01:00" />
  </head>
  <body>
    <header><p>Search, menu, and language selector</p></header>
    <main>
      <h1 id="main">Work immigration</h1>
      <p>You need to choose the right application route before you apply.</p>
      <h2 id="skilled-workers">Skilled workers</h2>
      <p>Skilled workers normally need a job offer and relevant qualifications.</p>
      <ul>
        <li>The employer must usually provide details about the position.</li>
      </ul>
      <h2>Seasonal workers</h2>
      <p>Seasonal work has its own route and is normally temporary.</p>
    </main>
    <footer><p>Contact, privacy, and accessibility</p></footer>
  </body>
</html>
"""


def test_parse_official_page_extracts_section_snapshot_metadata() -> None:
    parsed = parse_official_page(
        FetchedPage(url="https://www.udi.no/en/want-to-apply/work-immigration/", html=SAMPLE_HTML),
        language="en",
    )

    assert parsed.title == "Work immigration"
    assert parsed.canonical_url == "https://www.udi.no/en/want-to-apply/work-immigration/"
    assert parsed.language == "en"
    assert parsed.official_last_updated_at == datetime(2026, 1, 15, 8, 30, tzinfo=UTC)
    assert parsed.content_hash
    assert [section.heading for section in parsed.sections] == [
        "Work immigration",
        "Skilled workers",
        "Seasonal workers",
    ]
    assert parsed.sections[1].url.endswith("#skilled-workers")
    assert "Contact, privacy" not in parsed.raw_text


def test_parse_official_page_hash_is_stable_for_same_html() -> None:
    first = parse_official_page(
        FetchedPage(url="https://www.udi.no/en/want-to-apply/work-immigration/", html=SAMPLE_HTML),
        language="en",
    )
    second = parse_official_page(
        FetchedPage(url="https://www.udi.no/en/want-to-apply/work-immigration/", html=SAMPLE_HTML),
        language="en",
    )

    assert first.content_hash == second.content_hash
    assert [section.text for section in first.sections] == [section.text for section in second.sections]


def test_long_sections_are_split_with_same_source_url_and_heading() -> None:
    parsed = parse_official_page(
        FetchedPage(
            url="https://www.udi.no/en/want-to-apply/work-immigration/",
            html="<main><h1 id='long'>Long section</h1><p>" + "word " * 920 + "</p></main>",
        ),
        language="en",
    )

    chunks = list(chunk_section(parsed.sections[0]))

    assert len(chunks) == 3
    assert chunks[0].heading == "Long section (part 1)"
    assert chunks[1].heading == "Long section (part 2)"
    assert all(chunk.url.endswith("#long") for chunk in chunks)
    assert all(len(chunk.text.split()) <= 450 for chunk in chunks)


def test_invalid_numeric_last_updated_dates_are_ignored() -> None:
    assert _parse_date_value("31-13-2026") is None


def test_external_canonical_url_falls_back_to_fetched_url() -> None:
    parsed = parse_official_page(
        FetchedPage(
            url="https://www.udi.no/en/want-to-apply/work-immigration/",
            html="""
            <html>
              <head><link rel="canonical" href="https://example.com/mirror" /></head>
              <body><main><h1 id="main">Work</h1><p>Use the official UDI route before applying today.</p></main></body>
            </html>
            """,
        ),
        language="en",
        allowed_hosts=frozenset({"udi.no", "www.udi.no"}),
    )

    assert parsed.canonical_url == "https://www.udi.no/en/want-to-apply/work-immigration/"
    assert parsed.sections[0].url == "https://www.udi.no/en/want-to-apply/work-immigration/#main"


def test_section_anchor_is_url_encoded() -> None:
    parsed = parse_official_page(
        FetchedPage(
            url="https://www.udi.no/en/want-to-apply/work-immigration/",
            html="<main><h1 id='work route'>Work</h1><p>Use the official UDI route before applying today.</p></main>",
        ),
        language="en",
    )

    assert parsed.sections[0].url.endswith("#work%20route")
