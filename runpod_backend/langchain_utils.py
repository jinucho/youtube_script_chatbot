import asyncio
import logging
import os
import warnings

import tiktoken
from config import settings
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def calculate_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


warnings.filterwarnings(action="ignore")

logger = logging.getLogger(__name__)

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT or ""
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT or ""

MAX_TOKENS = 4096


class LangChainService:
    _instances = {}  # session_id별로 LangChainService 인스턴스를 저장

    @classmethod
    def get_instance(cls, session_id: str):
        if session_id not in cls._instances:
            cls._instances[session_id] = cls(session_id)
        return cls._instances[session_id]

    def __init__(self, session_id: str):
        self.session_id = session_id  # session_id를 인스턴스 변수로 저장
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=10
        )
        self.summarize_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=500
        )
        self.partial_summary_prompt = hub.pull(
            "teddynote/summary-stuff-documents-korean"
        )
        self.final_summary_prompt = self.partial_summary_prompt.copy()
        self.partial_summary_prompt.template = settings.PARTIAL_SUMMARY_PROMPT_TEMPLATE
        self.final_summary_prompt.template = settings.FINAL_SUMMARY_PROMPT_TEMPLATE
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, streaming=True)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.retriever = None
        self.is_prepared = False
        self.SUMMARY_RESULT = ""
        self.documents = []

    async def summarize(self, transcript: dict):
        # 각 세션별 요약 내용 및 상태 초기화
        self.documents = [
            Document(page_content="\n".join([t["text"] for t in transcript["script"]]))
        ]
        total_tokens = calculate_tokens(self.documents[0].page_content)
        # Create stuff documents chain for summarization
        partial_summary_chain = create_stuff_documents_chain(
            self.llm, self.partial_summary_prompt
        )
        final_summary_chain = create_stuff_documents_chain(
            self.llm, self.final_summary_prompt
        )
        if total_tokens > MAX_TOKENS:
            split_docs = self.summarize_splitter.split_documents(self.documents)
            print("문서 분할 완료")
            partial_summaries = []
            for split_doc in split_docs:
                partial_summary = await partial_summary_chain.ainvoke(
                    {"context": [split_doc]}
                )
                partial_summaries.append(partial_summary)
            print("분할 요약 완료")
            # Execute the summary chain
            partial_summaries_doc = [
                Document(page_content="\n".join(partial_summaries))
            ]
            self.SUMMARY_RESULT = await final_summary_chain.ainvoke(
                {"context": partial_summaries_doc}
            )
            print("최종 요약 완료")
            # Prepare the retriever after summarization
            await self.prepare_retriever()

            return self.SUMMARY_RESULT
        else:
            self.SUMMARY_RESULT = await partial_summary_chain.ainvoke(
                {"context": self.documents}
            )
            print("최종 요약 완료")
            await self.prepare_retriever()
            return self.SUMMARY_RESULT

    async def prepare_retriever(self):
        if self.is_prepared:
            return

        # Add summary to the first document
        try:
            for line in self.SUMMARY_RESULT.strip().split("\n"):
                self.documents[0].page_content += "\n" + line

            split_docs = self.text_splitter.split_documents(self.documents)
            print(f"Split_docs = {split_docs[0]}")
            vec_store = FAISS.from_documents(split_docs, self.embeddings)
            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = 10
            vec_retriever = vec_store.as_retriever(search_kwargs={"k": 10})

            # 세션별로 Retriever 설정
            self.retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vec_retriever],
                weights=[0.7, 0.3],
            )
            self.is_prepared = True
        except Exception as e:
            logger.error(
                f"Error preparing retriever for session {self.session_id}: {e}"
            )
            raise e

    async def stream_chat(self, prompt: str):
        if not self.is_prepared:
            await self.prepare_retriever()

        try:
            chat_prompt = PromptTemplate.from_template(
                """당신은 유튜브 스크립트 기반의 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
                당신의 임무는 주어진 영상의 텍스트 문맥(context)과 내부 지식을 활용하여 주어진 질문(question)에 답하는 것입니다.

                1. 검색된 다음 문맥(context)을 사용하여 질문(question)에 답하세요. 
                2. 영상과 관련 없는 질문일 경우 "영상과 관계 없는 질문 입니다." 라고 답하세요.
                3. 만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, 내부 지식(internal knowledge)을 사용하여 답변을 생성하세요.
                4. 만약, 문맥에서 답을 찾을 수 없고 내부 지식으로도 답변할 수 없다면, `주어진 정보에서 질문에 대한 답변을 찾을 수 없습니다`라고 답하세요.
                5. 내부 지식은 검토 후 답변하세요.
                6. 한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

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
            async for chunk in chain.astream(prompt):
                yield (
                    chunk
                    if isinstance(chunk, str)
                    else getattr(chunk, "content", str(chunk))
                )
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Error streaming for session {self.session_id}: {e}")
            raise e
