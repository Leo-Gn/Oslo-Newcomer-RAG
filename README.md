# Oslo Newcomer Assistant

A RAG assistant for newcomers to Oslo navigating public services like UDI, NAV, Skatteetaten, and Oslo kommune.

## One-Time Source Snapshot

The source registry is kept in `sources.yml` and only official allowlisted pages are fetched. To collect the static snapshot, start Postgres, apply migrations, make sure `DATABASE_URL` is exported in the shell, then run:

```bash
uv run oslo-ingest-sources
```

The command stores source metadata, page text, section headings, section URLs, collection time, detected official update time when available, and stable text hashes. Running the command again with unchanged pages skips the existing documents instead of duplicating chunks.

The app reads stored snapshot metadata from:

```bash
GET /api/sources
```
