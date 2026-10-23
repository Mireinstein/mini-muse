# My Muse

A personal AI assistant that answers visitors' questions about me. Go give it a try at [admiremadyira.com](https://admiremadyira.com). Also deployed at [my-muse-wine.vercel.app](https://my-muse-wine.vercel.app).

## How it works

Everything it knows about me is a few KB of text: my resume, a short about note,
and my public GitHub repos. That fits comfortably in the model's context, so
there is no retrieval step. Each question goes to a single serverless function,
`api/chat.py`, which sends the whole corpus, the conversation so far, and the
question to `gpt-oss-20b` on Groq and returns the answer.

The system prompt restricts answers to what is in the corpus. Anything outside it
gets pointed to the resume or LinkedIn instead of a guess, and a visitor's message
is treated as a question to answer, never as an instruction that changes the
rules.

The corpus is baked into `api/corpus.txt`. `build_corpus.py` regenerates it from
the live resume PDF, the GitHub API, and `docs/about.md`, so updating what the
assistant knows is one script run and a push.

## Update the corpus

```bash
pip install pypdf
python build_corpus.py
git add api/corpus.txt && git commit -m "Refresh the corpus" && git push
```

Vercel redeploys on push.

## Tech stack

- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Backend:** Python serverless function on Vercel
- **Model:** `gpt-oss-20b` via the Groq API
