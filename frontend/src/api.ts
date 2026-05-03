import type { ChatHistoryMessage, ChatResponse, SourceSnapshot, UiLanguage } from "./types";

export async function fetchSourceSnapshot(): Promise<SourceSnapshot> {
  const response = await fetch("/api/sources");
  if (!response.ok) {
    throw new Error("Could not load source snapshot");
  }
  return response.json();
}

export async function askQuestion(
  question: string,
  uiLanguage: UiLanguage,
  sessionHistory: ChatHistoryMessage[]
): Promise<ChatResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      question,
      ui_language: uiLanguage,
      session_history: sessionHistory
    })
  });

  if (!response.ok) {
    const detail = await readErrorDetail(response);
    throw new Error(detail);
  }

  return response.json();
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    return "The assistant could not answer right now";
  }

  return "The assistant could not answer right now";
}
