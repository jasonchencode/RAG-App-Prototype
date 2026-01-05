import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc


load_dotenv()

# Initialize Inngest client for background workflows
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

# Ingest a PDF by chunking, embedding, and storing it in the vector database
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    # Load a PDF and split it into text chunks
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    # Embed chunks and upsert them into Qdrant
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        # UUID v5 ensures there are no duplicates
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))


    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

# Answer a question using retrieved document context
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)

    # Search the vector database for relevant chunks
    def _search(question: str, top_k: int) -> RAGSearchResult:
        query_embedding = embed_texts([question])[0]
        storage = QdrantStorage()
        results = storage.search(query_embedding, top_k=top_k)
        return RAGSearchResult(contexts=results["contexts"], sources=results["sources"])

    # Generate a ChatGPT response solely from retrieved context
    def _generate_answer(search_result: RAGSearchResult, question: str) -> RAGQueryResult:
        contexts = search_result.contexts
        sources = search_result.sources
        
        if not contexts:
            return RAGQueryResult(
                answer="I couldn't find any relevant information in the documents to answer your question.",
                sources=[],
                num_contexts=0
            )
        
        # Build context string
        context_str = "\n\n---\n\n".join(contexts)
        
        # Generate answer using OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. "
                               "Only use information from the context to answer. If the context doesn't contain "
                               "enough information to answer the question, say so."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\n---\n\nQuestion: {question}"
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Respond with most similar vector (top choice)
        answer = response.choices[0].message.content
        return RAGQueryResult(answer=answer, sources=sources, num_contexts=len(contexts))

    search_result = await ctx.step.run(
        "search-documents",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult
    )
    
    result = await ctx.step.run(
        "generate-answer",
        lambda: _generate_answer(search_result, question),
        output_type=RAGQueryResult
    )
    
    return result.model_dump()


app = FastAPI()

# Health check endpoint
@app.get("/")
def health():
    return {"status": "ok"}

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])
