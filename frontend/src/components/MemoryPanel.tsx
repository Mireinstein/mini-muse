import { useEffect, useState } from "react";
import { forgetMemory, getMemories } from "../api";
import type { Memory } from "../types";

// What the agent remembers about the user, with a "forget" control per item —
// mirrors Muse letting you see and edit what it has learned.
export default function MemoryPanel() {
  const [memories, setMemories] = useState<Memory[]>([]);

  async function refresh() {
    try {
      setMemories(await getMemories());
    } catch {
      /* backend momentarily busy; keep what we have */
    }
  }

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, []);

  async function forget(id: string) {
    await forgetMemory(id);
    setMemories((m) => m.filter((x) => x.id !== id));
  }

  return (
    <div className="border-t border-slate-800 p-4">
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-400">
        What I remember
      </h2>
      {memories.length === 0 ? (
        <div className="text-sm text-slate-500">
          Nothing yet. Tell me a preference and I'll remember it.
        </div>
      ) : (
        <ul className="space-y-1">
          {memories.map((m) => (
            <li
              key={m.id}
              className="group flex items-start justify-between gap-2 rounded-lg border border-slate-700 bg-slate-800/50 px-3 py-2 text-sm text-slate-300"
            >
              <span>{m.text}</span>
              <button
                onClick={() => forget(m.id)}
                title="Forget this"
                className="shrink-0 text-slate-500 opacity-0 transition hover:text-rose-400 group-hover:opacity-100"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
