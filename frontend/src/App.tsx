import Chat from "./components/Chat";

export default function App() {
  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1">
        <Chat />
      </div>
    </div>
  );
}
