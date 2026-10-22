import type { ChatMessage } from "./types";

// Ask one question. Sends the recent turns so follow-ups keep their referent,
// and returns the full answer. The serverless function is fast enough that a
// plain request/response (no streaming) keeps things simple and reliable.
export async function ask(message: string, history: ChatMessage[]): Promise<string> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `Request failed (${res.status})`);
  return data.answer as string;
}
