import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Clock,
  Database,
  ExternalLink,
  Globe2,
  Loader2,
  Send,
  ShieldAlert
} from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useMemo, useState } from "react";

import { askQuestion, fetchSourceSnapshot } from "./api";
import type { ChatHistoryMessage, ChatTurn, SourceSnapshot, UiLanguage } from "./types";

const text = {
  en: {
    appName: "Oslo Newcomer Assistant",
    subtitle: "Official-source answers for practical first steps in Norway.",
    language: "Language",
    examples: [
      "What should I do after moving to Oslo?",
      "How do I get a tax deduction card?",
      "Where can students find housing support?"
    ],
    placeholder: "Ask about UDI, NAV, tax, Oslo services, SUA, or SiO",
    send: "Send",
    thinking: "Checking sources",
    snapshot: "Static source snapshot",
    chatTitle: "New conversation",
    prompts: "Suggested questions",
    sources: "sources",
    chunks: "chunks",
    collected: "Collected",
    updated: "Updated",
    unavailable: "Source snapshot unavailable",
    noDate: "not listed",
    citations: "Sources",
    refusal: "Could not answer safely",
    disclaimer: "Note",
    answer: "Answer",
    question: "You",
    error: "Request failed",
    empty: "Ask a practical question about moving to Oslo or Norway.",
    official: "Official source",
    mobileCheck: "Ready"
  },
  no: {
    appName: "Oslo Newcomer Assistant",
    subtitle: "Svar fra offentlige kilder om praktiske første steg i Norge.",
    language: "Språk",
    examples: [
      "Hva bør jeg gjøre etter at jeg flytter til Oslo?",
      "Hvordan får jeg skattekort?",
      "Hvor kan studenter finne hjelp med bolig?"
    ],
    placeholder: "Spør om UDI, NAV, skatt, Oslo-tjenester, SUA eller SiO",
    send: "Send",
    thinking: "Sjekker kilder",
    snapshot: "Statisk kildeutdrag",
    chatTitle: "Ny samtale",
    prompts: "Forslag",
    sources: "kilder",
    chunks: "tekstbiter",
    collected: "Hentet",
    updated: "Oppdatert",
    unavailable: "Kildeutdrag er ikke tilgjengelig",
    noDate: "ikke oppgitt",
    citations: "Kilder",
    refusal: "Kunne ikke svare trygt",
    disclaimer: "Merk",
    answer: "Svar",
    question: "Du",
    error: "Forespørselen feilet",
    empty: "Spør om praktiske steg ved flytting til Oslo eller Norge.",
    official: "Offentlig kilde",
    mobileCheck: "Klar"
  }
} as const;

function App() {
  const [language, setLanguage] = useState<UiLanguage>("en");
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [snapshot, setSnapshot] = useState<SourceSnapshot | null>(null);
  const [snapshotError, setSnapshotError] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const copy = text[language];
  const history = useMemo(() => buildHistory(turns), [turns]);

  useEffect(() => {
    let active = true;

    fetchSourceSnapshot()
      .then((data) => {
        if (active) {
          setSnapshot(data);
          setSnapshotError(false);
        }
      })
      .catch(() => {
        if (active) {
          setSnapshotError(true);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  async function submitQuestion(prompt: string) {
    const trimmed = prompt.trim();
    if (!trimmed || loading) {
      return;
    }

    setQuestion("");
    setError(null);
    setLoading(true);

    try {
      const response = await askQuestion(trimmed, language, history);
      setTurns((current) => [
        ...current,
        {
          id: response.answer_id || makeId(),
          question: trimmed,
          response
        }
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : copy.error);
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion(question);
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      void submitQuestion(question);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="window-dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>

        <div className="brand-block">
          <p className="eyebrow">Oslo RAG demo</p>
          <h1>{copy.appName}</h1>
          <p>{copy.subtitle}</p>
        </div>

        <div className="topbar-actions">
          <SnapshotPill snapshot={snapshot} hasError={snapshotError} language={language} />
          <LanguageSwitch language={language} onChange={setLanguage} label={copy.language} />
        </div>
      </header>

      <main className="workspace">
        <section className="chat-surface" data-testid="chat-surface" aria-label={copy.answer}>
          <div className="chat-head">
            <div className="assistant-mark">
              <BookOpen aria-hidden="true" className="h-5 w-5" />
            </div>
            <div>
              <p>{copy.appName}</p>
              <h2>{copy.chatTitle}</h2>
            </div>
          </div>

          <div className="prompt-card">
            <div className="prompt-title">{copy.prompts}</div>
            <div className="prompt-strip" aria-label="Example prompts">
              {copy.examples.map((example) => (
                <button
                  className="example-button"
                  disabled={loading}
                  key={example}
                  type="button"
                  onClick={() => void submitQuestion(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          <div className="message-stage">
            {turns.length === 0 ? (
              <div className="empty-state">
                <BookOpen aria-hidden="true" className="h-5 w-5" />
                <span>{copy.empty}</span>
              </div>
            ) : (
              <div className="space-y-8" data-testid="chat-history">
                {turns.map((turn) => (
                  <ChatExchange copy={copy} key={turn.id} language={language} turn={turn} />
                ))}
              </div>
            )}
          </div>

          {error ? (
            <div className="error-card" role="alert">
              <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
              <div>
                <p className="font-semibold">{copy.error}</p>
                <p className="mt-1">{error}</p>
              </div>
            </div>
          ) : null}

          {loading ? (
            <div className="loading-row" role="status">
              <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
              <span>{copy.thinking}</span>
            </div>
          ) : null}

          <form className="composer" onSubmit={onSubmit}>
            <textarea
              aria-label={copy.placeholder}
              className="composer-input"
              disabled={loading}
              maxLength={2000}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder={copy.placeholder}
              rows={3}
              value={question}
            />
            <button className="send-button" disabled={loading || !question.trim()} type="submit">
              <Send aria-hidden="true" className="h-4 w-4" />
              <span>{copy.send}</span>
            </button>
          </form>
        </section>

        <aside className="source-panel" aria-label={copy.snapshot}>
          <div className="panel-title">
            <Database aria-hidden="true" className="h-5 w-5" />
            <h2>{copy.snapshot}</h2>
          </div>
          <SourceSnapshotDetails copy={copy} hasError={snapshotError} language={language} snapshot={snapshot} />
        </aside>
      </main>
    </div>
  );
}

function LanguageSwitch({
  language,
  label,
  onChange
}: {
  language: UiLanguage;
  label: string;
  onChange: (language: UiLanguage) => void;
}) {
  return (
    <div className="language-switch" aria-label={label}>
      <Globe2 aria-hidden="true" className="h-4 w-4 text-fjord" />
      {(["en", "no"] as const).map((option) => (
        <button
          aria-pressed={language === option}
          className={language === option ? "language-button language-button-active" : "language-button"}
          key={option}
          type="button"
          onClick={() => onChange(option)}
        >
          {option.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

function SnapshotPill({
  snapshot,
  hasError,
  language
}: {
  snapshot: SourceSnapshot | null;
  hasError: boolean;
  language: UiLanguage;
}) {
  const copy = text[language];
  if (hasError || !snapshot || !snapshot.database_configured) {
    return (
      <div className="snapshot-pill">
        <Clock aria-hidden="true" className="h-4 w-4" />
        <span>{copy.unavailable}</span>
      </div>
    );
  }

  return (
    <div className="snapshot-pill" data-testid="snapshot-pill">
      <Clock aria-hidden="true" className="h-4 w-4" />
      <span>
        {snapshot.total_sources} {copy.sources}, {snapshot.total_chunks} {copy.chunks}
      </span>
    </div>
  );
}

function SourceSnapshotDetails({
  copy,
  hasError,
  language,
  snapshot
}: {
  copy: (typeof text)[UiLanguage];
  hasError: boolean;
  language: UiLanguage;
  snapshot: SourceSnapshot | null;
}) {
  if (hasError || !snapshot || !snapshot.database_configured) {
    return <p className="snapshot-empty">{copy.unavailable}</p>;
  }

  const collectedAt = latestDate(snapshot.sources.map((source) => source.collected_at));
  const updatedAt = latestDate(snapshot.sources.map((source) => source.official_last_updated_at));

  return (
    <div className="snapshot-details">
      <dl className="metric-grid">
        <div className="metric">
          <dt>{copy.sources}</dt>
          <dd>{snapshot.total_sources}</dd>
        </div>
        <div className="metric">
          <dt>{copy.chunks}</dt>
          <dd>{snapshot.total_chunks}</dd>
        </div>
      </dl>

      <dl className="date-stack">
        <DateLine label={copy.collected} language={language} value={collectedAt} />
        <DateLine label={copy.updated} language={language} value={updatedAt} />
      </dl>

      <div className="source-list">
        {snapshot.sources.slice(0, 6).map((source) => (
          <a className="source-link" href={source.url} key={source.url} rel="noreferrer" target="_blank">
            <span>
              {source.owner} <span>/{source.category}</span>
            </span>
            <ExternalLink aria-hidden="true" className="h-4 w-4 shrink-0" />
          </a>
        ))}
      </div>
    </div>
  );
}

function ChatExchange({
  copy,
  language,
  turn
}: {
  copy: (typeof text)[UiLanguage];
  language: UiLanguage;
  turn: ChatTurn;
}) {
  const response = turn.response;

  return (
    <article className="space-y-4">
      <div className="question-row">
        <span className="speaker-label">{copy.question}</span>
        <p>{turn.question}</p>
      </div>

      <div className={response.refused ? "answer-block answer-refusal" : "answer-block"}>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          {response.refused ? (
            <span className="state-badge state-badge-refusal">
              <ShieldAlert aria-hidden="true" className="h-4 w-4" />
              {copy.refusal}
            </span>
          ) : (
            <span className="state-badge state-badge-answer">
              <CheckCircle2 aria-hidden="true" className="h-4 w-4" />
              {copy.answer}
            </span>
          )}
        </div>

        <div className="answer-text">
          {response.answer.split("\n").map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>

        {response.disclaimer ? (
          <div className="disclaimer" data-testid="disclaimer">
            <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <p className="font-semibold">{copy.disclaimer}</p>
              <p>{response.disclaimer}</p>
            </div>
          </div>
        ) : null}

        <dl className="answer-dates">
          <DateLine label={copy.collected} language={language} value={response.data_currency.collected_at} />
          <DateLine label={copy.updated} language={language} value={response.data_currency.official_last_updated_at} />
        </dl>
      </div>

      {response.citations.length > 0 ? (
        <div>
          <h3 className="citation-heading">{copy.citations}</h3>
          <div className="citation-grid" data-testid="citation-list">
            {response.citations.map((citation) => (
              <a
                className="citation-card"
                href={citation.section_url || citation.source_url}
                key={citation.citation_id}
                rel="noreferrer"
                target="_blank"
              >
                <span className="citation-kicker">{citation.citation_id}</span>
                <span className="font-semibold">{citation.source_owner}</span>
                <span className="citation-section">{citation.section_heading}</span>
                <span className="citation-link-label">
                  {copy.official}
                  <ExternalLink aria-hidden="true" className="h-3.5 w-3.5" />
                </span>
              </a>
            ))}
          </div>
        </div>
      ) : null}
    </article>
  );
}

function DateLine({
  label,
  language,
  value
}: {
  label: string;
  language: UiLanguage;
  value: string | null;
}) {
  return (
    <div className="date-line">
      <dt>{label}</dt>
      <dd>{formatDate(value, language)}</dd>
    </div>
  );
}

function buildHistory(turns: ChatTurn[]): ChatHistoryMessage[] {
  return turns.slice(-6).flatMap((turn) => [
    { role: "user" as const, content: turn.question },
    { role: "assistant" as const, content: turn.response.answer }
  ]);
}

function latestDate(values: (string | null)[]): string | null {
  const timestamps = values
    .filter((value): value is string => Boolean(value))
    .map((value) => Date.parse(value))
    .filter((timestamp) => Number.isFinite(timestamp));

  if (timestamps.length === 0) {
    return null;
  }

  return new Date(Math.max(...timestamps)).toISOString();
}

function formatDate(value: string | null, language: UiLanguage) {
  if (!value) {
    return text[language].noDate;
  }

  return new Intl.DateTimeFormat(language === "no" ? "nb-NO" : "en-GB", {
    year: "numeric",
    month: "short",
    day: "numeric"
  }).format(new Date(value));
}

function makeId() {
  if ("randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `turn-${Date.now()}`;
}

export default App;
