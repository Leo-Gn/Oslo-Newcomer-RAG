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
  Sun,
  ThumbsDown,
  ThumbsUp
} from "lucide-react";
import { FormEvent, KeyboardEvent, MouseEvent, useEffect, useMemo, useRef, useState } from "react";

import { askQuestion, fetchSourceSnapshot, submitFeedback } from "./api";
import type { ChatHistoryMessage, ChatTurn, FeedbackRating, SourceSnapshot, UiLanguage } from "./types";

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
      "Where can students find housing support?",
      "How do I get a national identity number or D number?",
      "Where can I check residence card information?",
      "What should EU or EEA citizens check before working in Norway?",
      "Where can I read about family immigration?",
      "What should skilled workers check before applying?",
      "Where can I check information about permanent residence?",
      "Where can I check citizenship rules?",
      "How can I find a GP or healthcare information in Oslo?",
      "Where can I learn Norwegian in Oslo?",
      "What should families check after moving to Oslo?",
      "Where can I book an appointment as a foreign worker?"
    ],
    placeholder: "Type your question here...",
    send: "Send",
    thinking: "Checking official sources",
    prompts: "Suggested questions",
    clear: "Clear chat",
    reset: "Start over",
    dataUpdated: "Updated data",
    noDate: "not listed",
    citations: "Sources",
    copy: "Copy",
    copied: "Copied",
    refusal: "Could not answer safely",
    disclaimer: "Note",
    answer: "Answer",
    question: "You",
    error: "Request failed",
    empty: "Ask about permits, public services, work, tax, housing, or international student resources.",
    helpful: "Helpful",
    notHelpful: "Not helpful",
    feedbackSaved: "Feedback saved",
    feedbackFailed: "Feedback failed"
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
      "Hvor kan studenter finne hjelp med bolig?",
      "Hvordan får jeg fødselsnummer eller D-nummer?",
      "Hvor kan jeg sjekke informasjon om oppholdskort?",
      "Hva bør EU- eller EØS-borgere sjekke før de jobber i Norge?",
      "Hvor kan jeg lese om familieinnvandring?",
      "Hva bør faglærte arbeidstakere sjekke før de søker?",
      "Hvor kan jeg sjekke informasjon om permanent opphold?",
      "Hvor kan jeg sjekke regler om statsborgerskap?",
      "Hvordan finner jeg fastlege eller helseinformasjon i Oslo?",
      "Hvor kan jeg lære norsk i Oslo?",
      "Hva bør familier sjekke etter flytting til Oslo?",
      "Hvor kan jeg bestille time som utenlandsk arbeidstaker?"
    ],
    placeholder: "Skriv spørsmålet ditt her...",
    send: "Send",
    thinking: "Sjekker offentlige kilder",
    prompts: "Forslag",
    clear: "Tøm chat",
    reset: "Start på nytt",
    dataUpdated: "Oppdaterte data",
    noDate: "ikke oppgitt",
    citations: "Kilder",
    copy: "Kopier",
    copied: "Kopiert",
    refusal: "Kunne ikke svare trygt",
    disclaimer: "Merk",
    answer: "Svar",
    question: "Du",
    error: "Forespørselen feilet",
    empty: "Spør om opphold, offentlige tjenester, arbeid, skatt, bolig eller ressurser for internasjonale studenter.",
    helpful: "Nyttig",
    notHelpful: "Ikke nyttig",
    feedbackSaved: "Tilbakemelding lagret",
    feedbackFailed: "Tilbakemelding feilet"
  }
} as const;

const VISIBLE_EXAMPLE_COUNT = 3;
const MAX_COMPOSER_HEIGHT = 168;

type FeedbackStatus = {
  rating: FeedbackRating | null;
  pending: boolean;
  failed: boolean;
};

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
  const [feedbackByAnswer, setFeedbackByAnswer] = useState<Record<string, FeedbackStatus>>({});
  const [error, setError] = useState<string | null>(null);
  const activeRequestRef = useRef<AbortController | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const messageStageRef = useRef<HTMLDivElement | null>(null);
  const requestVersionRef = useRef(0);

  const copy = text[language];
  const history = useMemo(() => buildHistory(turns), [turns]);
  const examples = useMemo(() => selectExamplePrompts(copy.examples), [language]);
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
          language,
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
    setFeedbackByAnswer({});
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

  async function sendTurnFeedback(turn: ChatTurn, rating: FeedbackRating) {
    const answerId = turn.response.answer_id;
    const current = feedbackByAnswer[answerId];
    if (current?.pending) {
      return;
    }
    const nextRating: FeedbackRating = current?.rating === rating ? 0 : rating;
    const nextState = nextRating === 0 ? null : nextRating;

    setFeedbackByAnswer((statuses) => ({
      ...statuses,
      [answerId]: { rating: current?.rating ?? null, pending: true, failed: false }
    }));

    try {
      await submitFeedback(answerId, nextRating, citationChunkIds(turn.response.citations));
      setFeedbackByAnswer((statuses) => ({
        ...statuses,
        [answerId]: { rating: nextState, pending: false, failed: false }
      }));
    } catch {
      setFeedbackByAnswer((statuses) => ({
        ...statuses,
        [answerId]: { rating: current?.rating ?? null, pending: false, failed: true }
      }));
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
              <span className="brand-title">
                <span className="brand-mark">
                  <BookOpen aria-hidden="true" className="h-3.5 w-3.5" />
                </span>
                <h1>{copy.appName}</h1>
              </span>
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
            <div className="chat-body">
              <button className="floating-clear-button" type="button" onClick={clearChat}>
                <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
                <span>{copy.clear}</span>
              </button>

              <div className="message-stage" ref={messageStageRef}>
                <div className="chat-history space-y-8" data-testid="chat-history">
                  {turns.map((turn) => (
                    <ChatExchange
                      copiedMessageId={copiedMessageId}
                      key={turn.id}
                      feedbackStatus={feedbackByAnswer[turn.response.answer_id]}
                      onFeedback={(rating) => void sendTurnFeedback(turn, rating)}
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
            </div>
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
                  {examples.map((example) => (
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
  copy,
  loading,
  onChange,
  onKeyDown,
  onSubmit,
  question,
  refCallback
}: {
  copy: (typeof text)[UiLanguage];
  loading: boolean;
  onChange: (value: string) => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  question: string;
  refCallback: (element: HTMLTextAreaElement | null) => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    resizeComposer(textareaRef.current);
  }, [question]);

  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        aria-label={copy.placeholder}
        className="composer-input"
        maxLength={2000}
        onChange={(event) => {
          onChange(event.target.value);
          resizeComposer(event.target);
        }}
        onKeyDown={onKeyDown}
        placeholder={copy.placeholder}
        ref={(element) => {
          textareaRef.current = element;
          refCallback(element);
          resizeComposer(element);
        }}
        rows={1}
        value={question}
      />
      <div className="composer-actions">
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
      <Globe2 aria-hidden="true" className="h-4 w-4" />
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
  feedbackStatus,
  onFeedback,
  onCopy,
  turn
}: {
  copiedMessageId: string | null;
  feedbackStatus?: FeedbackStatus;
  onFeedback: (rating: FeedbackRating) => void;
  onCopy: (id: string, value: string) => void;
  turn: ChatTurn;
}) {
  const response = turn.response;
  const copy = text[turn.language];
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
          <div className="message-actions">
            <CopyButton
              copied={copiedMessageId === answerCopyId}
              copyLabel={copy.copy}
              copiedLabel={copy.copied}
              onClick={() => onCopy(answerCopyId, response.answer)}
            />
            <FeedbackButtons
              copy={copy}
              failed={feedbackStatus?.failed ?? false}
              onFeedback={onFeedback}
              pending={feedbackStatus?.pending ?? false}
              rating={feedbackStatus?.rating ?? null}
            />
          </div>
        </div>

        <AnswerBody answer={response.answer} />

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

function FeedbackButtons({
  copy,
  failed,
  onFeedback,
  pending,
  rating
}: {
  copy: (typeof text)[UiLanguage];
  failed: boolean;
  onFeedback: (rating: FeedbackRating) => void;
  pending: boolean;
  rating: FeedbackRating | null;
}) {
  const statusLabel = failed ? copy.feedbackFailed : rating ? copy.feedbackSaved : undefined;

  return (
    <div className="feedback-controls" aria-label={statusLabel}>
      <button
        aria-label={copy.helpful}
        aria-pressed={rating === 1}
        className={rating === 1 ? "feedback-button feedback-button-active" : "feedback-button"}
        disabled={pending}
        title={copy.helpful}
        type="button"
        onClick={() => onFeedback(1)}
      >
        <ThumbsUp aria-hidden="true" className="h-3.5 w-3.5" />
      </button>
      <button
        aria-label={copy.notHelpful}
        aria-pressed={rating === -1}
        className={rating === -1 ? "feedback-button feedback-button-active" : "feedback-button"}
        disabled={pending}
        title={copy.notHelpful}
        type="button"
        onClick={() => onFeedback(-1)}
      >
        <ThumbsDown aria-hidden="true" className="h-3.5 w-3.5" />
      </button>
    </div>
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

function AnswerBody({ answer }: { answer: string }) {
  const blocks = parseAnswerBlocks(answer);
  return (
    <div className="answer-text">
      {blocks.map((block, index) =>
        block.type === "list" ? (
          <ul key={index}>
            {block.items.map((item, itemIndex) => (
              <li key={itemIndex}>{item}</li>
            ))}
          </ul>
        ) : (
          <p key={index}>{block.text}</p>
        )
      )}
    </div>
  );
}

type AnswerBlock = { type: "paragraph"; text: string } | { type: "list"; items: string[] };

function parseAnswerBlocks(answer: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  let paragraph: string[] = [];
  let list: string[] = [];

  function flushParagraph() {
    if (paragraph.length) {
      blocks.push({ type: "paragraph", text: paragraph.join(" ") });
      paragraph = [];
    }
  }

  function flushList() {
    if (list.length) {
      blocks.push({ type: "list", items: list });
      list = [];
    }
  }

  for (const rawLine of answer.split("\n")) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }
    const bullet = line.match(/^[-*]\s+(.*)$/);
    if (bullet) {
      flushParagraph();
      list.push(bullet[1]);
    } else {
      flushList();
      paragraph.push(line);
    }
  }

  flushParagraph();
  flushList();
  return blocks;
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

function selectExamplePrompts(examples: readonly string[]) {
  const shuffled = [...examples];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled.slice(0, VISIBLE_EXAMPLE_COUNT);
}

function resizeComposer(element: HTMLTextAreaElement | null) {
  if (!element) {
    return;
  }

  element.style.height = "auto";
  const nextHeight = Math.min(element.scrollHeight, MAX_COMPOSER_HEIGHT);
  element.style.height = `${nextHeight}px`;
  element.style.overflowY = element.scrollHeight > MAX_COMPOSER_HEIGHT ? "auto" : "hidden";
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

function citationChunkIds(citations: ChatTurn["response"]["citations"]) {
  return Array.from(new Set(citations.map((citation) => citation.chunk_id)));
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
