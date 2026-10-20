import type { Approval } from "../types";

// Shown when Sentinel holds a write action for a human decision. Approve or Deny
// posts back to the server, which unblocks the paused tool call.
export default function ApprovalPrompt({
  approval,
  onDecide,
}: {
  approval: Approval;
  onDecide: (allow: boolean) => void;
}) {
  return (
    <div className="my-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3">
      <div className="mb-2 text-sm text-amber-200">
        <span className="font-semibold">Sentinel</span> needs your approval:
        <div className="mt-1 font-mono text-amber-100">{approval.description}</div>
      </div>
      <div className="flex gap-2">
        <button
          onClick={() => onDecide(true)}
          className="rounded bg-emerald-600 px-3 py-1 text-sm font-medium text-white hover:bg-emerald-500"
        >
          Approve
        </button>
        <button
          onClick={() => onDecide(false)}
          className="rounded bg-slate-700 px-3 py-1 text-sm font-medium text-slate-200 hover:bg-slate-600"
        >
          Deny
        </button>
      </div>
    </div>
  );
}
