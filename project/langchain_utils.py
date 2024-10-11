from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from config import settings
from dotenv import load_dotenv

load_dotenv()


if settings.LANGCHAIN_TRACING_V2:
    import os

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT or ""
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY or ""
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT or ""


class LangChainService:
    def __init__(self, settings):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=10
        )
        self.summary_prompt = hub.pull("teddynote/summary-stuff-documents-korean")
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            streaming=True,
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.retriever = None

    async def summarize(self, transcript: dict):
        documents = self.text_splitter.create_documents(
            [t["text"] for t in transcript["script"]]
        )
        summary_chain = self.summary_prompt | self.llm
        summary = await summary_chain.ainvoke({"context": documents})
        return summary

    async def prepare_retriever(self, transcript: dict):
        documents = self.text_splitter.create_documents(
            [t["text"] for t in transcript["script"]]
        )
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )

    async def stream_chat(self, prompt: str):
        if not self.retriever:
            raise ValueError("Retriever is not prepared. Call prepare_retriever first.")

        qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
        )

        async for chunk in qa_chain.astream({"question": prompt}):
            if "answer" in chunk:
                yield chunk["answer"]
