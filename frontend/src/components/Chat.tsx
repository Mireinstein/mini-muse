import { useEffect, useRef, useState } from "react";
import { chatStream, getHistory, postApprove } from "../api";
import type { Approval, ChatEvent, ChatMessage } from "../types";
import Message from "./Message";
import ApprovalPrompt from "./ApprovalPrompt";

// Keep the same conversation id across reloads so persisted history comes back.
function loadConversationId(): string {
  try {
    const saved = localStorage.getItem("mini-muse-conversation");
    if (saved) return saved;
    const id = crypto.randomUUID();
    localStorage.setItem("mini-muse-conversation", id);
    return id;
  } catch {
    return crypto.randomUUID(); // private mode / blocked storage: in-memory only
  }
}

// Apply one streamed event to the in-flight assistant message (always the last).
function applyEvent(messages: ChatMessage[], ev: ChatEvent): ChatMessage[] {
  const copy = messages.slice();
  const last = { ...copy[copy.length - 1] };
  last.tools = last.tools.slice();

  switch (ev.type) {
    case "token":
      last.content += ev.text;
      break;
    case "answer":
      last.content = ev.content;
      break;
    case "tool_call":
      last.tools.push({ name: ev.name, args: ev.args });
      break;
    case "tool_result":
      for (let i = last.tools.length - 1; i >= 0; i--) {
        if (last.tools[i].name === ev.name && last.tools[i].result === undefined) {
          last.tools[i] = { ...last.tools[i], result: ev.result };
          break;
        }
      }
      break;
    case "error":
      last.content = (last.content ? last.content + "\n\n" : "") + `⚠️ ${ev.message}`;
      break;
    default:
      return messages; // usage / approval_request / done: no bubble change
  }
  copy[copy.length - 1] = last;
  return copy;
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [approval, setApproval] = useState<Approval | null>(null);
  const conversationId = useRef(loadConversationId());
  const bottomRef = useRef<HTMLDivElement>(null);

  // On first load, restore any persisted messages for this conversation.
  useEffect(() => {
    getHistory(conversationId.current)
      .then((msgs) =>
        setMessages(msgs.map((m) => ({ role: m.role, content: m.content, tools: [] }))),
      )
      .catch(() => {}); // no history yet / offline: start empty
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, approval]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setMessages((m) => [
      ...m,
      { role: "user", content: text, tools: [] },
      { role: "assistant", content: "", tools: [] },
    ]);
    setBusy(true);
    try {
      for await (const ev of chatStream(conversationId.current, text)) {
        setMessages((m) => applyEvent(m, ev));
        if (ev.type === "approval_request") {
          setApproval({ id: ev.id, description: ev.description });
        }
      }
    } catch (e) {
      setMessages((m) => applyEvent(m, { type: "error", message: String(e) }));
    } finally {
      setBusy(false);
      setApproval(null);
    }
  }

  async function decide(allow: boolean) {
    if (!approval) return;
    await postApprove(approval.id, allow);
    setApproval(null);
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="mt-10 text-center text-slate-500">
            Ask Mini Muse to search the web, read your docs, run a calculation, or
            remember a task.
          </div>
        )}
        {messages.map((m, i) => (
          <Message key={i} message={m} />
        ))}
        {approval && <ApprovalPrompt approval={approval} onDecide={decide} />}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-slate-800 p-3">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Message Mini Muse…"
            className="flex-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-slate-100 placeholder-slate-500 focus:border-sky-500 focus:outline-none"
          />
          <button
            onClick={send}
            disabled={busy}
            className="rounded-lg bg-sky-600 px-4 py-2 font-medium text-white hover:bg-sky-500 disabled:opacity-50"
          >
            {busy ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
