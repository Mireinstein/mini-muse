import Chat from "./components/Chat";

export default function App() {
  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-slate-800 px-4 py-3">
        <div className="mx-auto w-full max-w-2xl">
          <h1 className="text-lg font-semibold text-slate-100">
            Ask Admire <span className="text-slate-500">· about Admire Madyira</span>
          </h1>
        </div>
      </header>
      <div className="min-h-0 flex-1">
        <Chat />
      </div>
    </div>
  );
}
