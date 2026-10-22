import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "../types";
import Citations, { extractCitations } from "./Citation";

// One chat bubble. User messages are plain text; assistant messages render
// Markdown and chip any source citations the answer carries.
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

  const citations = extractCitations(message.content);

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%]">
        {message.content && (
          <div className="prose-chat rounded-2xl rounded-bl-sm bg-slate-800 px-4 py-2 text-slate-100">
            <ReactMarkdown>{message.content}</ReactMarkdown>
            <Citations sources={citations} />
          </div>
        )}
      </div>
    </div>
  );
}
