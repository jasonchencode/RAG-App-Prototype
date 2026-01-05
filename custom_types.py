import pydantic


# Holds chunks and document identifiers
class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

# Reports how many chunks were ingested
class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

# Contains retrieved text contexts and their sources
class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

# Represents the final RAG answer and supporting metadata
class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
    