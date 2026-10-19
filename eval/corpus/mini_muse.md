# Mini Muse

Mini Muse is a tool-using agent. It calls the Groq API with the gpt-oss-20b model.
For retrieval it splits documents into 800-character chunks with 100 characters of overlap.
Plans and their steps are stored in SQLite so a run survives being killed and restarted.
