// The agent cites a source inline as e.g. [notes.md #3]. We pull those tags out
// of an answer and show them as chips under the message.

const CITATION_RE = /\[([^\]]+?#\d+)\]/g;

export function extractCitations(text: string): string[] {
  const found = new Set<string>();
  for (const match of text.matchAll(CITATION_RE)) {
    found.add(match[1].trim());
  }
  return [...found];
}

export default function Citations({ sources }: { sources: string[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {sources.map((s) => (
        <span
          key={s}
          className="rounded-full bg-sky-500/15 px-2 py-0.5 text-xs text-sky-300 ring-1 ring-sky-500/30"
        >
          {s}
        </span>
      ))}
    </div>
  );
}
