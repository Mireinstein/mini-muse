import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../types";

const CALENDLY_URL = "https://calendly.com/admiremadyira";

function BookingCalendar() {
  const params = new URLSearchParams({
    embed_type: "Inline",
    embed_domain: window.location.hostname,
    hide_gdpr_banner: "1",
  });
  return (
    <iframe
      src={`${CALENDLY_URL}?${params}`}
      title="Book a time with Admire"
      className="mt-2 h-[660px] w-full rounded-xl border border-slate-700 bg-white"
    />
  );
}

// One chat bubble. User messages are plain text; assistant messages render Markdown.
export default function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-sky-600 px-4 py-2 text-white">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-start">
      <div className="prose-chat max-w-[85%] rounded-2xl rounded-bl-sm bg-slate-800 px-4 py-2 text-slate-100">
        {/* remark-gfm turns the bare URLs the model writes into links; new tab
            because the chat usually runs inside an iframe on the site */}
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
          }}
        >
          {message.content}
        </ReactMarkdown>
      </div>
      {message.sources && message.sources.length > 0 && (
        <div className="mt-1.5 flex max-w-[85%] flex-wrap gap-1.5 pl-1">
          {message.sources.map((s) => (
            <span
              key={s}
              className="rounded-full border border-slate-700 px-2 py-0.5 text-xs text-slate-400"
              title="Source this answer is based on"
            >
              {s}
            </span>
          ))}
        </div>
      )}
      {message.booking && <BookingCalendar />}
    </div>
  );
}
