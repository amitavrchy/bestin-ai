# Bestin AI

Internal FastAPI service for book ingestion, grounded question answering, speech and avatar lip sync. The product backend calls this service on port 8001 with `X-Internal-Token`; see [the shared API contract](docs/API_CONTRACT.md).

PDF pages, EPUB chapters and text/Markdown chapters are cleaned and split into overlapping passages. A background job embeds them in batches and stores one cosine Chroma collection per book. Job status is persisted as JSON under `data/`; interrupted jobs are marked failed at startup and can be re-ingested. Questions retrieve passages and supply numbered sources and recent conversation history to an LLM. If all LLM providers fail, a short, cited extractive answer is returned. Suggested questions are generated from fixed, general-purpose templates. Embedding model changes require re-ingestion. Speech strips citation markers and markup before synthesis, caches results under `data/cache`, and uses Edge word timings, optionally Rhubarb, and text-based fallbacks. The ingestion worker pool runs at most two jobs concurrently. Pollinations requests are spaced at least 30 seconds apart per process; busy requests fall through to extractive answers.

## Setup

On macOS arm64, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12, then:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
cp .env.example .env
make run
```

The first local embedding call downloads the model to `data/models`; allow network access and disk space. Run `scripts/setup.sh` to install dependencies, Rhubarb, and warm the embedding model; alternatively run `scripts/install_rhubarb.sh` to install only Rhubarb. Run `scripts/dev.sh` for the development server. Its macOS release may require Rosetta on Apple Silicon. Audio conversion uses the imageio-ffmpeg bundled binary; a system ffmpeg is not required.

## Providers and configuration

| Provider | Purpose | Key |
| --- | --- | --- |
| FastEmbed | Local embeddings | None; model downloaded on first use |
| Edge TTS | Neural speech and word boundaries | None |
| Pollinations | Test LLM | None; subject to strict rate limits and availability |
| Groq | LLM and Whisper transcription | [Free key](https://console.groq.com/keys) |
| Gemini | LLM and optional embeddings | [Google AI Studio key](https://aistudio.google.com/apikey) |
| OpenAI | Optional LLM, embeddings, transcription and speech | [API key](https://platform.openai.com/api-keys) |

Set the shared `AI_INTERNAL_TOKEN` to the same value as the backend. `LLM_PROVIDER` defaults to `auto`; configured Groq, Gemini and OpenAI keys are tried in that order, followed by Pollinations and extractive fallback. Set it to a provider name to prioritize that provider. `EMBEDDING_PROVIDER` selects `fastembed`, `openai`, or `gemini`; `EMBEDDING_MODEL` selects the local model. `STT_PROVIDER` is `auto`, `groq`, or `openai`; without a key the endpoint returns `stt_unavailable`. `TTS_PROVIDER` is `edge` or `openai`. `LIPSYNC_BACKEND` is `auto`, `rhubarb`, `words`, or `heuristic`. `DATA_DIR` controls local storage and `REQUEST_TIMEOUT` controls provider HTTP timeouts. Model and base URL settings for each chat provider can also be overridden through their corresponding environment variables in `app/core/config.py`. See [.env.example](.env.example).

## Tests

```sh
make test
python3 .github/scripts/malware-check.py
```

## Troubleshooting

A failed ingestion includes an error in its status response. Image-only PDFs require OCR before upload. If a model download fails, check network access and retry ingestion. If transcription is unavailable, configure a supported key or use the frontend's browser speech recognition. Edge TTS requires network access; speech errors return 503 from `/v1/tts` and a null speech field on `/v1/ask`. If Rhubarb is unavailable or fails, lip sync falls back to word timings or a text heuristic. Do not expose this internal service directly to browsers; protect the shared token in deployment.

## Verified

On macOS arm64 on 2026-11-19, a keyless local smoke test ingested the Alice excerpt in `tests/fixtures/sample.txt` using FastEmbed. The ask endpoint returned a Pollinations answer with Edge MP3 audio and Rhubarb visemes. `/v1/tts` returned audio with Rhubarb visemes, `/v1/voices` returned 200, and `/v1/stt` returned 503 without a key. The smoke server was stopped and the test book was deleted. The automated suite has 44 passing tests.
