import asyncio
import logging
import os
import warnings
from operator import itemgetter

import tiktoken
from config import settings
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


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
    _instances = {}
    _message_store = {}  # 세션별 대화 기록 저장소

    @classmethod
    def get_instance(cls, session_id: str):
        if session_id not in cls._instances:
            cls._instances[session_id] = cls(session_id)
        return cls._instances[session_id]

    def __init__(self, session_id: str):
        self.session_id = session_id
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
        self.retriever = None
        self.is_prepared = False
        self.SUMMARY_RESULT = ""
        self.documents = []
        self._setup_rag_prompt()

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """세션 ID를 기반으로 대화 기록을 가져오는 함수"""
        if session_id not in self._message_store:
            self._message_store[session_id] = ChatMessageHistory()
        return self._message_store[session_id]

    def _setup_rag_prompt(self):
        """RAG 체인 설정"""
        self.prompt = PromptTemplate.from_template(
            """당신은 유튜브 스크립트 기반의 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다.
                당신의 주요 임무는 다음과 같습니다:
                    1. 기본적으로 검색된 문맥(context)과 이전 대화 내용(chat_history)을 바탕으로 질문에 답변하세요.
                    
                    2. 다음과 같은 경우에는 자연스럽게 내부 지식을 활용하여 답변하세요:
                        - 검색된 문맥이 질문의 의도를 완벽하게 충족하지 못할 때
                        - 영상의 전반적인 주제와 연관되지만 구체적인 답변이 문맥에 없을 때
                        - 문맥에서 부분적인 정보만 찾을 수 있을 때는 문맥의 정보와 내부 지식을 조합하여 답변하세요

                    3. 답변 시 다음 사항을 지켜주세요:
                    - 항상 자연스러운 대화체로 답변하세요
                    - 문맥에서 답을 찾을 수 없더라도, 그 사실을 언급하지 말고 바로 답변하세요
                    - 기술 용어나 고유명사는 원어를 유지하세요
                    - 전문적인 내용도 이해하기 쉽게 설명하세요

                    4. 만약 질문이 영상의 주제나 내용과 전혀 관련이 없다면 "영상과 관계 없는 질문입니다."라고 답변하세요.

                    #이전 대화 내용:
                    {chat_history}

                    #질문:
                    {question}

                    #문맥:
                    {context}

                    #답변:"""
        )

    async def summarize(self, transcript: dict):
        self.documents = [
            Document(page_content="\n".join([t["text"] for t in transcript["script"]]))
        ]
        total_tokens = calculate_tokens(self.documents[0].page_content)

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

            partial_summaries_doc = [
                Document(page_content="\n".join(partial_summaries))
            ]
            self.SUMMARY_RESULT = await final_summary_chain.ainvoke(
                {"context": partial_summaries_doc}
            )
            print("최종 요약 완료")
            await self.prepare_retriever()
            return self.SUMMARY_RESULT
        else:
            self.SUMMARY_RESULT = await final_summary_chain.ainvoke(
                {"context": self.documents}
            )
            print("최종 요약 완료")
            await self.prepare_retriever()
            return self.SUMMARY_RESULT

    async def prepare_retriever(self):
        if self.is_prepared:
            return

        try:
            for line in self.SUMMARY_RESULT.strip().split("\n"):
                self.documents[0].page_content += "\n" + line

            split_docs = self.text_splitter.split_documents(self.documents)
            print(f"Split_docs = {split_docs[0]}")
            vec_store = FAISS.from_documents(split_docs, self.embeddings)
            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = 10
            vec_retriever = vec_store.as_retriever(search_kwargs={"k": 10})

            self.retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vec_retriever],
                weights=[0.7, 0.3],
            )
            self.is_prepared = True

            # RAG 체인 설정
            self.chain = (
                {
                    "context": itemgetter("question") | self.retriever,
                    "question": itemgetter("question"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            # RunnableWithMessageHistory 설정
            self.chain_with_history = RunnableWithMessageHistory(
                self.chain,
                self._get_session_history,
                input_messages_key="question",
                history_messages_key="chat_history",
            )

        except Exception as e:
            logger.error(
                f"Error preparing retriever for session {self.session_id}: {e}"
            )
            raise e

    async def stream_chat(self, prompt: str):
        if not self.is_prepared:
            await self.prepare_retriever()

        try:
            async for chunk in self.chain_with_history.astream(
                {"question": prompt},
                config={"configurable": {"session_id": self.session_id}},
            ):
                content = (
                    chunk
                    if isinstance(chunk, str)
                    else getattr(chunk, "content", str(chunk))
                )
                yield content
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Error streaming for session {self.session_id}: {e}")
            raise e
