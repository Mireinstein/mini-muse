# My Muse

A personal AI assistant that answers visitors' questions about me. Go give it a try at [admiremadyira.com](https://admiremadyira.com). Also deployed at [my-muse-wine.vercel.app](https://my-muse-wine.vercel.app).

## How it works

What it knows about me comes from my resume, a short about note, and my public
GitHub repos. `build_corpus.py` pulls all three, splits them into chunks of up to
800 characters along their headings, and embeds each chunk once with Gemini's
`gemini-embedding-001`. The chunks and their vectors are committed as
`api/index.json`.

Each question goes to a single serverless function, `api/chat.py`. It embeds the
question, ranks the stored chunks by cosine similarity, and sends the top five to
`gpt-oss-20b` on Groq along with the conversation so far. A follow-up question is
searched together with the one before it, so "and when was that?" still finds the
right chunk. The index is small enough to rank in plain Python, so there is no
vector database to host.

The system prompt restricts answers to the retrieved text. Anything outside it gets
pointed to the resume or LinkedIn instead of a guess, and a visitor's message is
treated as a question to answer, never as an instruction that changes the rules.

## Update the corpus

```bash
pip install pypdf
python build_corpus.py   # needs GEMINI_API_KEY in .env
git add api/corpus.txt api/index.json && git commit -m "Refresh the corpus" && git push
```

Vercel redeploys on push.

## Tech stack

- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Backend:** Python serverless function on Vercel
- **Model:** `gpt-oss-20b` via the Groq API
- **Embeddings:** `gemini-embedding-001` via the Gemini API
