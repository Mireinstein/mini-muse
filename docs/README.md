# Your documents

Put the notes and PDFs you want Mini Muse to answer from in this folder —
`.md`, `.txt`, and `.pdf` files (nested folders are fine). Then index them:

```bash
python -m mini_muse --ingest
```

Once indexed, ask questions and the agent will search these documents and cite
the passage it used:

```bash
python -m mini_muse "What did I decide about X?"
```

This folder ships empty; the documents are yours to add.
