# StudyPilot

StudyPilot turns your notes into searchable study material: ask questions with cited sources, review flashcards, and take a quiz. Without database configuration, notes and study materials stay in the app session; there is no database requirement.

## Run locally

Use Python 3.13 to match the project's tested virtual environment and deployment configuration.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The app can read `OPENAI_API_KEY`, `APP_ACCESS_CODE`, and `DATABASE_URL` from environment variables or Streamlit secrets. For local secrets, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace only the values you use. Never commit the real secrets file.

```powershell
streamlit run app.py
```

The first time notes are indexed, StudyPilot downloads the `all-MiniLM-L6-v2` embedding model. The model is loaded on demand and cached for the running app. Ask, Flashcards, and Quiz need an OpenAI key; keyword and summary analysis and note search run locally. Uploads are limited to 5 MB, pasted notes to 50,000 characters, and model calls to 20 per session.

## Optional PostgreSQL persistence

Set `DATABASE_URL` to a PostgreSQL connection string in the environment or `.streamlit/secrets.toml`. For example:

```toml
DATABASE_URL = "postgresql://USER:PASSWORD@HOST:5432/DATABASE"
```

When the database is configured, create a private data ID in the sidebar and download it. Re-enter that ID in a later session to restore the associated documents, page-aware chunks, embeddings, flashcards, quiz, and quiz progress. The ID is a bearer credential: anyone with it can read or delete that saved data, so keep it private. StudyPilot stores only a hash of the ID in PostgreSQL.

The database enforces a limit of 20 model calls per data ID per UTC day, alongside the existing 20-call session limit. The daily limit is a usage guardrail, not an authentication or security control. **Delete my data** removes that ID's saved documents, study materials, quiz progress, and daily usage records from the configured database, and clears that session's workspace.

If `DATABASE_URL` is not set, no storage controls or database calls are used and the app continues in its current in-memory mode. If a user does not connect a private data ID, their current workspace also remains session-only.

## Privacy

When database persistence is active, StudyPilot stores the user's cleaned notes, chunks, embedding arrays, flashcards, quiz content and results, and daily call count in the configured PostgreSQL database. It does not log or print the database URL, private data ID, uploaded text, or pasted notes. Database encryption, backups, retention, and access controls depend on the PostgreSQL provider and its configuration.

Using Ask sends retrieved note chunks to OpenAI. Flashcards and Quiz send up to 6,000 characters of notes to OpenAI. Summary, keywords, indexing, and search run locally. Do not use persistence for notes you are not comfortable storing with your configured database provider.

## Tests

Run the local test suite with:

```powershell
pytest
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository, choose the branch, and set `app.py` as the entrypoint.
3. In **Advanced settings**, select Python 3.13 to match the local environment.
4. Add the real `OPENAI_API_KEY` and optional `APP_ACCESS_CODE` in the app's **Secrets** field. Add `DATABASE_URL` only if you want PostgreSQL persistence. Use the TOML format in `.streamlit/secrets.toml.example` as a template, replacing placeholders with real values.
5. Save and deploy.

`requirements.txt` is at the repository root and pins the app and test dependencies. The embedding model is downloaded on first use, so initial indexing can take longer than later indexing.

## Optional: CPU-only PyTorch

Sentence Transformers supports CPU inference; GPU-enabled PyTorch is not required for this app. To replace an existing PyTorch build with the CPU wheel, use the official CPU package index:

```powershell
python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
```

This command is optional and is not part of the deployment requirements. Select a Python and PyTorch version supported by the official CPU wheel index for the target environment.
