import logging
from app.schemas.requests import Ask
from app.providers.registry import chat
from app.providers.speech import synthesize
from app.services import library
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import auth, valid_id

router = APIRouter()

@router.post("/v1/ask", dependencies=[Depends(auth)])
async def ask(body: Ask):
    book_id = valid_id(str(body.book_id))
    record = library.status(book_id)
    if record["status"] == "not_found":
        raise HTTPException(404, "book_not_found")
    if record["status"] != "ready":
        raise HTTPException(409, "book_not_ready")
    try:
        coll = library.db().get_collection(library.collection_name(book_id))
        metadata = coll.metadata or {}
        if (
            metadata.get("embedding_model") != library.embeddings.model
            or metadata.get("embedding_provider") != library.embeddings.provider
        ):
            raise HTTPException(409, "embedding_model_changed_reingest_required")
        query = library.embeddings.embed([body.question])
        result = coll.query(
            query_embeddings=query, n_results=min(body.top_k, coll.count())
        )
    except HTTPException:
        raise
    except Exception:
        logging.exception("Retrieval failed")
        raise HTTPException(500, "retrieval_failed")
    citations = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        citations.append(
            {
                "chunk_id": chunk_id,
                "text": text[:600],
                "page": meta.get("page") or None,
                "chapter": meta.get("chapter") or None,
                "score": float(1 - distance),
            }
        )
    sources = "\n\n".join(
        f"[{i}] {text}" for i, text in enumerate(result["documents"][0], 1)
    )
    messages = [
        {
            "role": "system",
            "content": "Answer only from the numbered book sources. Cite factual claims with [1], [2], etc. If the answer is not in the sources, say you do not know. "
            + (
                "Keep the answer around 120 words or less."
                if body.style == "concise"
                else "Give a detailed answer."
            )
            + "\n\nSources:\n"
            + sources,
        }
    ]
    messages.extend(m.model_dump() for m in body.history[-6:])
    messages.append({"role": "user", "content": body.question})
    response = chat(messages)
    if response:
        answer, provider, model = response
    else:
        answer = (
            "I could not generate an answer. Here is a relevant passage from the book: "
            + citations[0]["text"]
            + " [1]"
            if citations
            else "I do not know based on this book."
        )
        provider, model = "extractive", "extractive"
    speech = None
    if body.speak:
        try:
            speech = await synthesize(answer, body.voice)
        except Exception:
            logging.exception("Speech synthesis failed")
    return {
        "answer": answer,
        "citations": citations,
        "provider": provider,
        "model": model,
        "speech": speech,
    }


