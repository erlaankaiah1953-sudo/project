# Restaurant RAG

## Run the chat frontend

From the project directory, start the Streamlit app with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app uses the existing Gemini RAG chain and Chroma database. If the vector
database has not been created yet, build it first:

```powershell
.\.venv\Scripts\python.exe ingest.py
```

Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the project `.env` file before
running the app.