import { useState } from "react";
import type { ToolActivity } from "../types";

// A collapsible card showing one tool the agent used: its arguments and result.
export default function ToolCall({ tool }: { tool: ToolActivity }) {
  const [open, setOpen] = useState(false);
  const pending = tool.result === undefined;

  let args = tool.args;
  try {
    args = JSON.stringify(JSON.parse(tool.args), null, 2);
  } catch {
    /* leave raw if it isn't valid JSON */
  }

  return (
    <div className="my-1 rounded-md border border-slate-700 bg-slate-800/60 text-sm">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-slate-300 hover:bg-slate-700/40"
      >
        <span className="text-slate-500">{open ? "▾" : "▸"}</span>
        <span className="font-mono text-sky-300">{tool.name}</span>
        <span className={pending ? "text-amber-400" : "text-emerald-400"}>
          {pending ? "running…" : "done"}
        </span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-slate-700 px-3 py-2">
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-500">arguments</div>
            <pre className="whitespace-pre-wrap break-words text-xs text-slate-300">{args}</pre>
          </div>
          {tool.result !== undefined && (
            <div>
              <div className="text-xs uppercase tracking-wide text-slate-500">result</div>
              <pre className="whitespace-pre-wrap break-words text-xs text-slate-300">
                {tool.result}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
