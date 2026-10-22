// Events streamed from POST /api/chat (Server-Sent Events).
export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "answer"; content: string }
  | { type: "error"; message: string }
  | { type: "done" };

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}
