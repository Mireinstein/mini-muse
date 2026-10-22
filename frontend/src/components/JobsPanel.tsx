import { useEffect, useRef, useState } from "react";
import { getPlan, getPlans, startGoal } from "../api";
import type { Plan, Step } from "../types";

const STATUS_COLOR: Record<string, string> = {
  pending: "text-slate-400",
  running: "text-amber-400",
  done: "text-emerald-400",
  failed: "text-rose-400",
};

export default function JobsPanel() {
  const [plans, setPlans] = useState<Plan[]>([]);
  const [goal, setGoal] = useState("");
  const [starting, setStarting] = useState(false);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const expandedRef = useRef<number | null>(null);
  expandedRef.current = expanded;

  // Poll the plan list (and the open plan's steps) so background progress is live.
  useEffect(() => {
    let active = true;
    async function tick() {
      try {
        const latest = await getPlans();
        if (active) setPlans(latest);
        const open = expandedRef.current;
        if (open !== null) {
          const detail = await getPlan(open);
          if (active) setSteps(detail.steps);
        }
      } catch {
        /* backend may be momentarily busy; try again next tick */
      }
    }
    tick();
    const timer = setInterval(tick, 2500);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  async function start() {
    const text = goal.trim();
    if (!text || starting) return;
    setStarting(true);
    try {
      await startGoal(text);
      setGoal("");
      setPlans(await getPlans());
    } finally {
      setStarting(false);
    }
  }

  async function toggle(id: number) {
    if (expanded === id) {
      setExpanded(null);
      setSteps([]);
      return;
    }
    setExpanded(id);
    const detail = await getPlan(id);
    setSteps(detail.steps);
  }

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-800 p-4">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-400">
          Background goals
        </h2>
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="Give Mini Muse a multi-step goal to work on in the background…"
          rows={3}
          className="w-full resize-none rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:border-sky-500 focus:outline-none"
        />
        <button
          onClick={start}
          disabled={starting}
          className="mt-2 w-full rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          {starting ? "Planning…" : "Start goal"}
        </button>
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto p-3">
        {plans.length === 0 && (
          <div className="mt-6 text-center text-sm text-slate-500">No goals yet.</div>
        )}
        {plans.map((plan) => (
          <div key={plan.id} className="rounded-lg border border-slate-700 bg-slate-800/50">
            <button
              onClick={() => toggle(plan.id)}
              className="flex w-full items-start justify-between gap-2 p-3 text-left"
            >
              <span className="text-sm text-slate-200">{plan.goal}</span>
              <span className={`shrink-0 text-xs ${STATUS_COLOR[plan.status] ?? "text-slate-400"}`}>
                {plan.status}
              </span>
            </button>
            {expanded === plan.id && (
              <ol className="space-y-1 border-t border-slate-700 p-3 text-sm">
                {steps.map((step) => (
                  <li key={step.id} className="flex gap-2">
                    <span className={`shrink-0 ${STATUS_COLOR[step.status] ?? "text-slate-400"}`}>
                      {step.status === "done" ? "✓" : step.idx + 1}
                    </span>
                    <span className="text-slate-300">{step.description}</span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
