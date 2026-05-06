export type UiLanguage = "en" | "no";

export type ChatHistoryMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatCitation = {
  citation_id: string;
  chunk_id: string;
  source_owner: string;
  source_url: string;
  section_url: string;
  section_heading: string;
  collected_at: string;
  official_last_updated_at: string | null;
};

export type ChatDataCurrency = {
  collected_at: string | null;
  official_last_updated_at: string | null;
};

export type ChatResponse = {
  answer_id: string;
  answer: string;
  refused: boolean;
  disclaimer: string | null;
  citations: ChatCitation[];
  data_currency: ChatDataCurrency;
};

export type FeedbackRating = -1 | 0 | 1;

export type FeedbackResponse = {
  feedback_id: string | null;
  created_at: string | null;
  cleared: boolean;
};

export type SourceSnapshot = {
  database_configured: boolean;
  total_sources: number;
  total_chunks: number;
  sources: SourceRow[];
};

export type SourceRow = {
  owner: string;
  url: string;
  language: string;
  category: string;
  intended_coverage: Record<string, unknown>;
  collected_at: string | null;
  official_last_updated_at: string | null;
  chunk_count: number;
};

export type ChatTurn = {
  id: string;
  language: UiLanguage;
  question: string;
  response: ChatResponse;
};
