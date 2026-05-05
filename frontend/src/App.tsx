import {
  AlertTriangle,
  ArrowUp,
  BookOpen,
  CheckCircle2,
  Clock,
  Copy,
  ExternalLink,
  Globe2,
  Moon,
  RotateCcw,
  ShieldAlert,
  Sun
} from "lucide-react";
import { FormEvent, KeyboardEvent, MouseEvent, useEffect, useMemo, useRef, useState } from "react";

import { askQuestion, fetchSourceSnapshot } from "./api";
import type { ChatHistoryMessage, ChatTurn, SourceSnapshot, UiLanguage } from "./types";

const text = {
  en: {
    appName: "Oslo Newcomer RAG",
    subtitle: "Navigating Norwegian bureaucracy with official sources.",
    language: "Language",
    themeLight: "Switch to light mode",
    themeDark: "Switch to dark mode",
    examples: [
      "What should I do after moving to Oslo?",
      "How do I get a tax deduction card?",
      "Where can students find housing support?"
    ],
    placeholder: "Ask about UDI, NAV, tax, work, housing, SUA, or SiO",
    send: "Send",
    thinking: "Checking official sources",
    prompts: "Suggested questions",
    sources: "sources",
    clear: "Clear chat",
    reset: "Start over",
    dataUpdated: "Updated data",
    unavailable: "Source snapshot unavailable",
    noDate: "not listed",
    citations: "Sources",
    copy: "Copy",
    copied: "Copied",
    refusal: "Could not answer safely",
    disclaimer: "Note",
    answer: "Answer",
    question: "You",
    error: "Request failed",
    empty: "Ask about permits, public services, work, tax, housing, or student life.",
    official: "Official source",
    mobileCheck: "Ready"
  },
  no: {
    appName: "Oslo Newcomer RAG",
    subtitle: "Hjelper deg med norsk byråkrati via offentlige kilder.",
    language: "Språk",
    themeLight: "Bytt til lys modus",
    themeDark: "Bytt til mørk modus",
    examples: [
      "Hva bør jeg gjøre etter at jeg flytter til Oslo?",
      "Hvordan får jeg skattekort?",
      "Hvor kan studenter finne hjelp med bolig?"
    ],
    placeholder: "Spør om UDI, NAV, skatt, arbeid, bolig, SUA eller SiO",
    send: "Send",
    thinking: "Sjekker offentlige kilder",
    prompts: "Forslag",
    sources: "kilder",
    clear: "Tøm chat",
    reset: "Start på nytt",
    dataUpdated: "Oppdaterte data",
    unavailable: "Kildeutdrag er ikke tilgjengelig",
    noDate: "ikke oppgitt",
    citations: "Kilder",
    copy: "Kopier",
    copied: "Kopiert",
    refusal: "Kunne ikke svare trygt",
    disclaimer: "Merk",
    answer: "Svar",
    question: "Du",
    error: "Forespørselen feilet",
    empty: "Spør om opphold, offentlige tjenester, arbeid, skatt, bolig eller studieliv.",
    official: "Offentlig kilde",
    mobileCheck: "Klar"
  }
} as const;

function App() {
  const [language, setLanguage] = useState<UiLanguage>("en");
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [snapshot, setSnapshot] = useState<SourceSnapshot | null>(null);
  const [snapshotError, setSnapshotError] = useState(false);
  const [loading, setLoading] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const activeRequestRef = useRef<AbortController | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const messageStageRef = useRef<HTMLDivElement | null>(null);
  const requestVersionRef = useRef(0);

  const copy = text[language];
  const history = useMemo(() => buildHistory(turns), [turns]);
  const dataUpdatedAt = useMemo(() => latestUpdateDate(snapshot, turns), [snapshot, turns]);
  const conversationStarted = turns.length > 0 || Boolean(pendingQuestion) || Boolean(error);

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
  }, [turns, pendingQuestion, loading]);

  async function submitQuestion(prompt: string) {
    const trimmed = prompt.trim();
    if (!trimmed || loading) {
      return;
    }

    setQuestion("");
    setError(null);
    setPendingQuestion(trimmed);
    setLoading(true);
    activeRequestRef.current?.abort();

    const controller = new AbortController();
    const requestVersion = requestVersionRef.current + 1;
    activeRequestRef.current = controller;
    requestVersionRef.current = requestVersion;

    try {
      const response = await askQuestion(trimmed, language, history, controller.signal);
      if (requestVersionRef.current !== requestVersion) {
        return;
      }
      setTurns((current) => [
        ...current,
        {
          id: response.answer_id || makeId(),
          question: trimmed,
          response
        }
      ]);
      setPendingQuestion(null);
    } catch (err) {
      if (requestVersionRef.current !== requestVersion || controller.signal.aborted) {
        return;
      }
      setError(err instanceof Error ? err.message : copy.error);
    } finally {
      if (requestVersionRef.current === requestVersion) {
        activeRequestRef.current = null;
        setLoading(false);
        requestAnimationFrame(() => composerRef.current?.focus());
      }
    }
  }

  function clearChat() {
    activeRequestRef.current?.abort();
    activeRequestRef.current = null;
    requestVersionRef.current += 1;
    setTurns([]);
    setQuestion("");
    setError(null);
    setPendingQuestion(null);
    setLoading(false);
    setCopiedMessageId(null);
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

  async function copyMessage(id: string, value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopiedMessageId(id);
      window.setTimeout(() => setCopiedMessageId((current) => (current === id ? null : current)), 1400);
    } catch {
      setCopiedMessageId(null);
    }
  }

  return (
    <div className={`app-shell theme-${theme}`}>
      <main className="workspace">
        <section
          className={conversationStarted ? "chat-surface chat-surface-active" : "chat-surface chat-surface-start"}
          data-testid="chat-surface"
          aria-label={copy.answer}
        >
          <header className="topbar">
            <button className="brand-block" type="button" onClick={clearChat} aria-label={copy.reset}>
              <h1>{copy.appName}</h1>
              <p>{copy.subtitle}</p>
            </button>

            <div className="topbar-actions">
              <ThemeSwitch
                label={theme === "dark" ? copy.themeLight : copy.themeDark}
                onToggle={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
                theme={theme}
              />
              <LanguageSwitch language={language} onChange={setLanguage} label={copy.language} />
            </div>
          </header>

          {conversationStarted ? (
            <>
              <div className="message-stage" ref={messageStageRef}>
                <div className="space-y-8" data-testid="chat-history">
                  {turns.map((turn) => (
                    <ChatExchange
                      copiedMessageId={copiedMessageId}
                      copy={copy}
                      key={turn.id}
                      language={language}
                      onCopy={(id, value) => void copyMessage(id, value)}
                      turn={turn}
                    />
                  ))}
                  {pendingQuestion ? (
                    <QuestionBubble
                      copied={copiedMessageId === "pending-question"}
                      copyLabel={copy.copy}
                      copiedLabel={copy.copied}
                      label={copy.question}
                      onCopy={() => void copyMessage("pending-question", pendingQuestion)}
                      question={pendingQuestion}
                    />
                  ) : null}
                  {loading ? <ThinkingRow text={copy.thinking} /> : null}
                </div>
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

              <Composer
                clearLabel={copy.clear}
                copy={copy}
                loading={loading}
                onChange={setQuestion}
                onClear={clearChat}
                onKeyDown={onComposerKeyDown}
                onSubmit={onSubmit}
                question={question}
                refCallback={(element) => {
                  composerRef.current = element;
                }}
              />
            </>
          ) : (
            <div className="start-screen" onClick={onMessageStageClick}>
              <div className="empty-state">
                <BookOpen aria-hidden="true" className="h-5 w-5" />
                <span>{copy.empty}</span>
              </div>

              <Composer
                copy={copy}
                loading={loading}
                onChange={setQuestion}
                onKeyDown={onComposerKeyDown}
                onSubmit={onSubmit}
                question={question}
                refCallback={(element) => {
                  composerRef.current = element;
                }}
              />

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
            </div>
          )}
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

function ThemeSwitch({
  label,
  onToggle,
  theme
}: {
  label: string;
  onToggle: () => void;
  theme: "light" | "dark";
}) {
  return (
    <button
      aria-label={label}
      aria-pressed={theme === "dark"}
      className="theme-switch"
      type="button"
      onClick={onToggle}
    >
      <span className={theme === "light" ? "theme-icon theme-icon-active" : "theme-icon"}>
        <Sun aria-hidden="true" className="h-4 w-4" />
      </span>
      <span className={theme === "dark" ? "theme-icon theme-icon-active" : "theme-icon"}>
        <Moon aria-hidden="true" className="h-4 w-4" />
      </span>
    </button>
  );
}

function Composer({
  clearLabel,
  copy,
  loading,
  onChange,
  onClear,
  onKeyDown,
  onSubmit,
  question,
  refCallback
}: {
  clearLabel?: string;
  copy: (typeof text)[UiLanguage];
  loading: boolean;
  onChange: (value: string) => void;
  onClear?: () => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  question: string;
  refCallback: (element: HTMLTextAreaElement | null) => void;
}) {
  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        aria-label={copy.placeholder}
        className="composer-input"
        maxLength={2000}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder={copy.placeholder}
        ref={refCallback}
        rows={2}
        value={question}
      />
      <div className="composer-actions">
        {onClear && clearLabel ? (
          <button
            aria-label={clearLabel}
            className="composer-icon-button composer-reset-button"
            type="button"
            onClick={onClear}
          >
            <RotateCcw aria-hidden="true" className="h-4 w-4" />
          </button>
        ) : null}
        <button className="send-button" disabled={loading || !question.trim()} type="submit">
          <ArrowUp aria-hidden="true" className="h-5 w-5" />
          <span>{copy.send}</span>
        </button>
      </div>
    </form>
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
  copiedMessageId,
  copy,
  language,
  onCopy,
  turn
}: {
  copiedMessageId: string | null;
  copy: (typeof text)[UiLanguage];
  language: UiLanguage;
  onCopy: (id: string, value: string) => void;
  turn: ChatTurn;
}) {
  const response = turn.response;
  const citations = compactCitations(response.citations);
  const questionCopyId = `${turn.id}-question`;
  const answerCopyId = `${turn.id}-answer`;

  return (
    <article className="space-y-4">
      <QuestionBubble
        copied={copiedMessageId === questionCopyId}
        copyLabel={copy.copy}
        copiedLabel={copy.copied}
        label={copy.question}
        onCopy={() => onCopy(questionCopyId, turn.question)}
        question={turn.question}
      />

      <div className={response.refused ? "answer-block answer-refusal" : "answer-block"}>
        <div className="message-toolbar">
          <div className="flex flex-wrap items-center gap-2">
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
          <CopyButton
            copied={copiedMessageId === answerCopyId}
            copyLabel={copy.copy}
            copiedLabel={copy.copied}
            onClick={() => onCopy(answerCopyId, response.answer)}
          />
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

function QuestionBubble({
  copied,
  copyLabel,
  copiedLabel,
  label,
  onCopy,
  question
}: {
  copied: boolean;
  copyLabel: string;
  copiedLabel: string;
  label: string;
  onCopy: () => void;
  question: string;
}) {
  return (
    <div className="question-row">
      <div className="question-meta">
        <span className="speaker-label">{label}</span>
        <CopyButton copied={copied} copyLabel={copyLabel} copiedLabel={copiedLabel} onClick={onCopy} />
      </div>
      <p>{question}</p>
    </div>
  );
}

function CopyButton({
  copied,
  copiedLabel,
  copyLabel,
  onClick
}: {
  copied: boolean;
  copiedLabel: string;
  copyLabel: string;
  onClick: () => void;
}) {
  return (
    <button className="copy-button" type="button" onClick={onClick}>
      <Copy aria-hidden="true" className="h-3.5 w-3.5" />
      <span>{copied ? copiedLabel : copyLabel}</span>
    </button>
  );
}

function ThinkingRow({ text }: { text: string }) {
  return (
    <div className="thinking-row" role="status">
      <span className="thinking-dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span>{text}</span>
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
