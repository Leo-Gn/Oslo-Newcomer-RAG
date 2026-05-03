# Oslo Newcomer Assistant

A RAG assistant for newcomers to Oslo navigating public services like UDI, NAV, Skatteetaten, and Oslo kommune.

## Local Development

Keep two VS Code terminals open while working on the app.

Terminal 1 runs the backend:

```bash
cd /home/leo/Desktop/Oslo-Newcomer-RAG
uv run oslo-newcomer-rag
```

The backend reads local settings from `.env`. Keep real keys and database passwords there only.

Terminal 2 runs the frontend:

```bash
cd /home/leo/Desktop/Oslo-Newcomer-RAG/frontend
npm run dev
```

Open the app at:

```text
http://127.0.0.1:5173
```

Stop either server with `Ctrl+C` in the terminal where it is running. If a terminal was closed and a server is still using a port, stop it from any project terminal:

```bash
lsof -ti:5173 | xargs -r kill
lsof -ti:8000 | xargs -r kill
```

The frontend is configured to use port `5173` only. If that port is busy, Vite now fails instead of silently moving to `5174` or `5175`.

## One-Time Source Snapshot

The source registry is kept in `sources.yml` and only official allowlisted pages are fetched. To collect the static snapshot, start Postgres, apply migrations, make sure `DATABASE_URL` is set in `.env`, then run:

```bash
uv run oslo-ingest-sources
```

The command stores source metadata, page text, section headings, section URLs, collection time, detected official update time when available, and stable text hashes. Running the command again with unchanged pages skips the existing documents instead of duplicating chunks.

The app reads stored snapshot metadata from:

```bash
GET /api/sources
```
