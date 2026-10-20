import Chat from "./components/Chat";
import JobsPanel from "./components/JobsPanel";

export default function App() {
  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-slate-800 px-4 py-3">
        <h1 className="text-lg font-semibold text-slate-100">
          Mini Muse <span className="text-slate-500">· personal assistant</span>
        </h1>
      </header>
      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[1fr_360px]">
        <Chat />
        <div className="hidden md:block">
          <JobsPanel />
        </div>
      </div>
    </div>
  );
}
