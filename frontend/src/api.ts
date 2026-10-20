import type { ChatEvent, Plan, Step } from "./types";

// Stream the agent's reply to one message. POST can't use EventSource, so we
// read the response body ourselves and parse the Server-Sent Events frames.
export async function* chatStream(
  conversationId: string,
  message: string,
): AsyncGenerator<ChatEvent> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, message }),
  });
  if (!res.body) throw new Error("No response stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (line) yield JSON.parse(line.slice(6)) as ChatEvent;
    }
  }
}

export async function postApprove(id: number, allow: boolean): Promise<void> {
  await fetch("/api/approve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, allow }),
  });
}

export async function startGoal(goal: string): Promise<{ plan_id: number; steps: string[] }> {
  const res = await fetch("/api/goals", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal }),
  });
  return res.json();
}

export async function getPlans(): Promise<Plan[]> {
  const res = await fetch("/api/plans");
  const data = await res.json();
  return data.plans as Plan[];
}

export async function getPlan(id: number): Promise<{ plan: Plan; steps: Step[] }> {
  const res = await fetch(`/api/plans/${id}`);
  return res.json();
}
