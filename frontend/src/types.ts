// Events streamed from POST /api/chat (Server-Sent Events).
export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "tool_call"; name: string; args: string }
  | { type: "tool_result"; name: string; result: string }
  | { type: "approval_request"; id: number; description: string }
  | { type: "usage"; prompt_tokens: number; completion_tokens: number }
  | { type: "answer"; content: string }
  | { type: "error"; message: string }
  | { type: "done" };

// One tool the assistant used during a turn, with its result once it returns.
export interface ToolActivity {
  name: string;
  args: string;
  result?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  tools: ToolActivity[];
}

// A pending Sentinel approval surfaced to the user.
export interface Approval {
  id: number;
  description: string;
}

// Background plan + steps from /api/plans.
export interface Step {
  id: number;
  idx: number;
  description: string;
  status: "pending" | "running" | "done" | "failed";
  result: string | null;
  error: string | null;
}

export interface Plan {
  id: number;
  goal: string;
  status: "running" | "done" | "failed";
  created_at: number;
}

// One thing the agent remembers about the user (from /api/memories).
export interface Memory {
  id: string;
  text: string;
}
