export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  booking?: boolean; // assistant asked to show the Calendly scheduler under this message
}
