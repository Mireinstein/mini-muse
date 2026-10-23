import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../types";

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
    <div className="flex justify-start">
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
    </div>
  );
}
