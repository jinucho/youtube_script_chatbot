import asyncio
import logging
import os

from config import settings
from dotenv import load_dotenv
from typing import List
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

# 임시 추가 라이브러리
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT or ""
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT or ""


# langchain_utils.py
class LangChainService:
    _instance = None  # Singleton instance를 위한 클래스 변수

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:  # 이미 초기화된 인스턴스라면 다시 초기화하지 않음
            return
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=10
        )
        self.summary_prompt = hub.pull("teddynote/summary-stuff-documents-korean")
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, streaming=True)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.retriever = None
        self.is_prepared = False
        self.SUMMARY_RESULT = ""
        self.documents = []
        self.__initialized = True
        logger.debug("LangChainService initialized")

    async def summarize(self, transcript: dict):
        self.documents = [
            Document(page_content="\n".join([t["text"] for t in transcript["script"]]))
        ]
        logger.debug(f"Documents created {self.documents}")

        # Create stuff documents chain for summarization
        summary_chain = create_stuff_documents_chain(self.llm, self.summary_prompt)

        # Execute the summary chain
        self.SUMMARY_RESULT = await summary_chain.ainvoke({"context": self.documents})
        logger.debug(f"Summary result in summarize: {self.SUMMARY_RESULT}")
        # Prepare the retriever after summarization
        await self.prepare_retriever()

        return self.SUMMARY_RESULT

    async def prepare_retriever(self):
        if self.is_prepared:
            logger.debug("Retriever already prepared")
            return
        # Add summary to the first document
        logger.debug("Adding summary to the first document")
        try:
            logger.debug(f"summary_result in prepare_retriever : {self.SUMMARY_RESULT}")
            for line in self.SUMMARY_RESULT.strip().split("\n"):
                self.documents[0].page_content += "\n" + line
            logger.debug(
                f"Summary added to the first document: {self.documents[0].page_content}"
            )
            logger.debug(f"no Docs: {len(self.documents)}")

            split_docs = self.text_splitter.split_documents(self.documents)

            logger.debug(f"Split docs: {len(split_docs)}")
            logger.debug(f"Split docs: {split_docs[0]}")
            logger.debug(
                f"split_docs 유효성 검증{self._validate_docs_for_faiss(split_docs)}"
            )
            vec_store = FAISS.from_documents(split_docs, self.embeddings)
            logger.debug("Vector store created")

            bm25_retriever = BM25Retriever.from_documents(split_docs)
            logger.debug("BM25 retriever created")
            bm25_retriever.k = 10
            logger.debug("BM25 retriever k set to 3")
            vec_retriever = vec_store.as_retriever(search_kwargs={"k": 10})
            logger.debug("Vector retriever created")
            self.retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vec_retriever],
                weights=[0.7, 0.3],
            )
            logger.debug("Ensemble retriever created")
            self.is_prepared = True
            logger.debug("Retriever prepared")
        except Exception as e:
            logger.error(f"Error preparing retriever: {e}")
            raise e

    def _validate_docs_for_faiss(self, docs: List[Document]):
        if not docs:
            raise ValueError("No documents provided for FAISS indexing")
        for i, doc in enumerate(docs):
            if not doc.page_content or not isinstance(doc.page_content, str):
                raise ValueError(f"Invalid document content at index {i}")
        logger.debug("Documents validated for FAISS indexing")

    async def stream_chat(self, prompt: str):
        logger.debug(f"Langchain Received prompt: {prompt}")
        logger.debug(f"리트리벌 준비 : {self.is_prepared}")
        logger.debug(f"Summary result in stream_chat: {self.SUMMARY_RESULT}")
        if not self.is_prepared:
            logger.debug("Retriever not prepared")
            await self.prepare_retriever()
            logger.debug("Retriever prepared")
            logger.debug(f"리트리벌 준비 : {self.is_prepared}")
            logger.debug(
                f"Summary result in stream_chat when self.is_prepared False : {self.SUMMARY_RESULT}"
            )
        try:
            chat_prompt = PromptTemplate.from_template(
                """당신은 유튜브 스크립트 기반의 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
                당신의 임무는 주어진 영상의 텍스트 문맥(context)과 내부 지식을 활용하여 주어진 질문(question)에 답하는 것입니다.

                1. 검색된 다음 문맥(context)을 사용하여 질문(question)에 답하세요. 
                2. 영상과 관련 없는 질문일 경우 "영상과 관계 없는 질문 입니다." 라고 답하세요.
                3. 만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, 내부 지식(internal knowledge)을 사용하여 답변을 생성하세요.
                4. 만약, 문맥에서 답을 찾을 수 없고 내부 지식으로도 답변할 수 없다면, `주어진 정보에서 질문에 대한 답변을 찾을 수 없습니다`라고 답하세요.
                5. 내부 지식은 검토 후 답변하세요.

                한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

                #Question: 
                {question} 

                #Context: 
                {context} 

                #Answer:"""
            )

            chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | chat_prompt
                | self.llm
                | StrOutputParser()
            )
            logger.debug("Starting streaming")
            async for chunk in chain.astream(prompt):
                if isinstance(chunk, str):
                    logger.debug(f"Streaming chunk: {chunk[:10]}")
                    yield chunk
                elif hasattr(chunk, "content"):
                    logger.debug(f"Streaming chunk: {chunk.content[:10]}")
                    yield chunk.content
                else:
                    logger.debug(f"Yielding other chunk type: {str(chunk)[:10]}...")
                    yield str(chunk)
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Error streaming: {e}")
            raise e
