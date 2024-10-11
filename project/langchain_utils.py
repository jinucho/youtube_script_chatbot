import asyncio
from config import settings
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

if settings.LANGCHAIN_TRACING_V2:
    import os

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT or ""
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY or ""
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT or ""


class LangChainService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=10
        )
        self.summary_prompt = hub.pull("teddynote/summary-stuff-documents-korean")
        self.summary_prompt.template = """Please summarize the sentence according to the following REQUEST.
REQUEST:
1. Summarize the main points in bullet points in KOREAN.
2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.
3. Use various emojis to make the summary more interesting.
4. Translate the summary into KOREAN if it is written in ENGLISH.
5. DO NOT translate any technical terms.
6. DO NOT include any unnecessary information.
7. Please refer to each summary and indicate the key topic.
8. If the original text is in English, we have already provided a summary translated into Korean, so please do not provide a separate translation.

CONTEXT:
{context}

SUMMARY:"""
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            streaming=True,
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.retriever = None
        self.SUMMARY_RESULT = ""
        self.DOCS = []

    async def summarize(self, transcript: dict):
        documents = self.text_splitter.create_documents(
            [t["text"] for t in transcript["script"]]
        )
        self.DOCS = documents

        # Create stuff documents chain for summarization
        summary_chain = create_stuff_documents_chain(self.llm, self.summary_prompt)

        # Execute the summary chain
        self.SUMMARY_RESULT = await summary_chain.ainvoke({"context": documents})

        # Prepare the retriever after summarization
        await self.prepare_retriever()

        return self.SUMMARY_RESULT

    async def prepare_retriever(self):
        # Add summary to the first document
        self.DOCS[0].page_content += "\n" + self.SUMMARY_RESULT.strip()

        split_docs = self.text_splitter.split_documents(self.DOCS)
        vec_store = FAISS.from_documents(split_docs, self.embeddings)
        bm25_retriever = BM25Retriever.from_documents(split_docs)
        bm25_retriever.k = 10
        vec_retriever = vec_store.as_retriever(search_kwargs={"k": 10})
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vec_retriever],
            weights=[0.7, 0.3],
        )

    async def stream_chat(self, prompt: str):
        if not self.retriever:
            await self.prepare_retriever()

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

        async for chunk in chain.astream(prompt):
            if isinstance(chunk, str):
                yield chunk
            elif hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)
            await asyncio.sleep(0)
