import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Clock,
  ExternalLink,
  Globe2,
  Loader2,
  RotateCcw,
  Send,
  ShieldAlert
} from "lucide-react";
import { FormEvent, KeyboardEvent, MouseEvent, useEffect, useMemo, useRef, useState } from "react";

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
    chatTitle: "New conversation",
    prompts: "Suggested questions",
    sources: "sources",
    clear: "Clear chat",
    reset: "Start over",
    dataUpdated: "Updated data",
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
    chatTitle: "Ny samtale",
    prompts: "Forslag",
    sources: "kilder",
    clear: "Tøm chat",
    reset: "Start på nytt",
    dataUpdated: "Oppdaterte data",
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
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const messageStageRef = useRef<HTMLDivElement | null>(null);

  const copy = text[language];
  const history = useMemo(() => buildHistory(turns), [turns]);
  const dataUpdatedAt = useMemo(() => latestUpdateDate(snapshot, turns), [snapshot, turns]);

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

  useEffect(() => {
    messageStageRef.current?.scrollTo({
      top: messageStageRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [turns, loading]);

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
      requestAnimationFrame(() => composerRef.current?.focus());
    }
  }

  function clearChat() {
    setTurns([]);
    setQuestion("");
    setError(null);
    requestAnimationFrame(() => composerRef.current?.focus());
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion(question);
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submitQuestion(question);
    }
  }

  function onMessageStageClick(event: MouseEvent<HTMLDivElement>) {
    const target = event.target as HTMLElement;
    if (target.closest("a,button")) {
      return;
    }
    composerRef.current?.focus();
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand-block" type="button" onClick={clearChat} aria-label={copy.reset}>
          <p className="eyebrow">Oslo RAG demo</p>
          <h1>{copy.appName}</h1>
          <p>{copy.subtitle}</p>
        </button>

        <div className="topbar-actions">
          <button className="clear-button" type="button" onClick={clearChat}>
            <RotateCcw aria-hidden="true" className="h-4 w-4" />
            <span>{copy.clear}</span>
          </button>
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

          {turns.length === 0 ? (
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
          ) : null}

          <div className="message-stage" onClick={onMessageStageClick} ref={messageStageRef}>
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
              maxLength={2000}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder={copy.placeholder}
              ref={composerRef}
              rows={3}
              value={question}
            />
            <button className="send-button" disabled={loading || !question.trim()} type="submit">
              <Send aria-hidden="true" className="h-4 w-4" />
              <span>{copy.send}</span>
            </button>
          </form>
        </section>
      </main>

      <footer className="data-footer" aria-live="polite">
        <Clock aria-hidden="true" className="h-3.5 w-3.5" />
        <span>
          {copy.dataUpdated}: {snapshotError ? copy.noDate : formatDate(dataUpdatedAt, language)}
        </span>
      </footer>
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
  const citations = compactCitations(response.citations);

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
      </div>

      {citations.length > 0 ? (
        <div className="citation-row" data-testid="citation-list">
          <span className="citation-heading">{copy.citations}</span>
          <div className="citation-pills">
            {citations.map((citation) => (
              <a
                aria-label={`${citation.source_owner}: ${citation.section_heading}`}
                className="citation-pill"
                href={citation.url}
                key={citation.url}
                rel="noreferrer"
                target="_blank"
              >
                <span className="font-semibold">{citation.source_owner}</span>
                <span>{citation.section_heading}</span>
                <ExternalLink aria-hidden="true" className="h-3.5 w-3.5" />
              </a>
            ))}
          </div>
        </div>
      ) : null}
    </article>
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

function latestUpdateDate(snapshot: SourceSnapshot | null, turns: ChatTurn[]): string | null {
  const snapshotDates = snapshot?.database_configured
    ? snapshot.sources.map((source) => source.official_last_updated_at)
    : [];
  const answerDates = turns.map((turn) => turn.response.data_currency.official_last_updated_at);

  return latestDate([...snapshotDates, ...answerDates]);
}

function compactCitations(citations: ChatTurn["response"]["citations"]) {
  const seen = new Map<string, (typeof citations)[number]>();

  for (const citation of citations) {
    const url = citation.section_url || citation.source_url;
    if (!seen.has(url)) {
      seen.set(url, citation);
    }
  }

  return Array.from(seen.values()).map((citation) => ({
    source_owner: citation.source_owner,
    section_heading: citation.section_heading,
    url: citation.section_url || citation.source_url
  }));
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
