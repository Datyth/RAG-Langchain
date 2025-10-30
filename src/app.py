import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from rag.base.llm_models import GeminiLLM
from rag.rag_system.main import build_rag_chain, InputQA, OutputQA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
llm = GeminiLLM(model_name = "gemini-2.5-flash", api_key = gemini_api_key)

genai_docs = "src/data_source/gen_ai"
genai_chain = build_rag_chain(llm, data_dir = genai_docs, data_type = "pdf")

app = FastAPI(
    title="RAG QA System with Gemini LLM",
    description="A Retrieval-Augmented (RAG) system using Gemini LLM for question-answering tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"])

@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/rag/genai", response_model = OutputQA)
async def rag_genai(input_data: InputQA) -> OutputQA:
    question = input_data.question
    answer = genai_chain.invoke({"question": question})
    return OutputQA(answer=answer)


add_routes(app,
    genai_chain,
    playground_type="default",
    path="/generative_ai")