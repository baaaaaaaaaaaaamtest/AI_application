from fastapi import FastAPI, Request, BackgroundTasks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import asyncio
import httpx
from langgraph.prebuilt import create_react_agent
import json
import requests
import bs4
import datetime
from langchain_teddynote import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_transformers import LongContextReorder
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader

import os

# API 키 정보 로드
load_dotenv()
logging.langsmith("CH99-kakao-Chatbot")

# OLLAMA_BASE_URL = "http://localhost:11434" # Ollama 서버 주소
# OLLAMA_CHAT_MODEL = "ollama-ko-0710:latest" # Ollama에서 실행중인 모델 이름


async def get_mcp_server_response(query):
    client = MultiServerMCPClient(
        {
            "naver-search-mcp": {
                "command": "node",
                "args": [
                    "/home/ansgyqja/naver-search-mcp/dist/src/index.js"
                ],
                "transport": "stdio",
                "cwd": "/home/ansgyqja/naver-search-mcp",
                "env": {
                    "NAVER_CLIENT_ID": os.getenv('NAVER_CLIENT_ID'),
                    "NAVER_CLIENT_SECRET": os.getenv('NAVER_CLIENT_SECRET')
                }
            }
        }
    )
    tools = await client.get_tools(server_name="naver-search-mcp")
    tool = next(tool for tool in tools if tool.name == "search_news")

    response = await tool.ainvoke({
        "query": query,
        "display": 100,  # 최대 100
        "start": 1,
        "sort": "date"  # 날짜순 정렬
    })
    data = json.loads(response)  # JSON 문자열을 dict로 변환
    return data



def get_news_origin(json_data):
    links_naver_sports = [ item["link"] for item in json_data["items"] if "https://m.sports.naver.com/" in item["link"]]
    links_naver_news = [ item["link"] for item in json_data["items"] if "https://n.news.naver.com/" in item["link"]]
    links_naver_entertain = [ item["link"] for item in json_data["items"] if "https://m.entertain.naver.com/" in item["link"]]
    sport_docs = []
    news_docs = []
    if len(links_naver_sports + links_naver_entertain) != 0:
        sport_loader = WebBaseLoader(
            web_paths=(links_naver_sports + links_naver_entertain),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    "div",
                    attrs={
                        "class": ["ArticleHead_article_head_title__YUNFf", "_article_content"]
                    },
                )
            ),
        )
        sport_docs = sport_loader.load()
    elif len(links_naver_news) != 0:
        news_loader = WebBaseLoader(
            web_paths=(links_naver_news),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    "div",
                    attrs={"class": ["media_end_head_title", "newsct_article _article_body"]},
                )
            ),
        )
        news_docs = news_loader.load()
    return sport_docs+news_docs
    

def get_split_docs(news_doc):
    # 1. 텍스트 청크 분할 (chunk_size=1000, chunk_overlap=50)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    # 2. Text loader
    # documents = [Document(page_content=result_text)]
    return text_splitter.split_documents(news_doc)

    # loader = TextLoader(result_text, encoding="utf-8")
    # return documents.load_and_split(text_splitter)

def get_retriever(split_docs): 
    model_name = "intfloat/multilingual-e5-large-instruct"
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},  # cuda, cpu
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.from_documents(documents=split_docs, embedding=hf_embeddings)
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.25, "fetch_k": 20}
    )
    return retriever


def get_bm25_retriever(split_docs):
    bm25_retriever = BM25Retriever.from_documents(
        split_docs
    )
    bm25_retriever.k = 10  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.
    return bm25_retriever


def get_reranker(retriever):
    reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=reranker, top_n=10)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

def reorder_documents(docs):
    # 재정렬
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    return reordered_docs

def get_prompt():
    template = """당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
    You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.
    모든 대답은 반드시 한국말로 대답해주세요.
    
    문서 내용:
    {context}
    질문: {question}
    답변:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt


def get_multi_query_retriever(retriever):
    # 다중 질의어 생성
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    multiquery_retriever = MultiQueryRetriever.from_llm(  # MultiQueryRetriever를 언어 모델을 사용하여 초기화합니다.
    # 벡터 데이터베이스의 retriever와 언어 모델을 전달합니다.
    retriever=retriever,
    llm=llm,
    )
    return multiquery_retriever


def get_esenmble_retriever(retriever1, retriever2):
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2],
        weights=[0.6, 0.4],  # 각 리트리버의 가중치를 설정합니다.
        k=10  # 최종적으로 반환할 문서의 개수를 설정합니다.
    )
    return ensemble_retriever




async def general_llm(news_doc,utterance):
    if len(news_doc) == 0:
        return "죄송해요. 뉴스를 찾지 못했어요. 다른 질문 해주실래요?"

    split_docs = get_split_docs(news_doc)

    faiss_retriever = get_retriever(split_docs)
    
    bm25_retriever = get_bm25_retriever(split_docs)

    faiss_multi_retriever = get_multi_query_retriever(faiss_retriever)

    bm25_multi_retriever = get_multi_query_retriever(bm25_retriever)

    esenmble_retriever = get_esenmble_retriever(faiss_multi_retriever, bm25_multi_retriever)

    compression_retriever = get_reranker(esenmble_retriever)
    
    prompt= get_prompt()
    
    
    # 7. Ollama LLM 초기화
    # llm = OllamaLLM(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL,temperature=1, max_tokens=1024)
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.7, max_tokens=1024)
    chain = (
    {
        "context": itemgetter("question")
        | compression_retriever
        | RunnableLambda(reorder_documents),  # 질문을 기반으로 문맥을 검색합니다.
        "question": itemgetter("question"),  # 질문을 추출합니다.
    }
    | prompt  # 프롬프트 템플릿에 값을 전달합니다.
    | llm # 언어 모델에 프롬프트를 전달합니다.
    | StrOutputParser()  # 모델의 출력을 문자열로 파싱합니다.
    )
    question = {"question": f'{utterance}'}
    answer = chain.invoke(question)
    return answer

def save_log(utterance,server_response,merge,responseBody):
    text_to_save = "\n\n".join(doc.page_content for doc in merge)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./query_log/process_log_{now}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("==== [질의 내용 utterance] ====\n")
        f.write(utterance + "\n\n")
        f.write("==== [get_news_origin 결과 data] ====\n")
        f.write(json.dumps(server_response, ensure_ascii=False, indent=2) + "\n\n")
        f.write("==== [크롤링 내용 result_text] ====\n")
        f.write(text_to_save)
        f.write("==== [responseBody 결과] ====\n")
        f.write(json.dumps(responseBody, ensure_ascii=False, indent=2) + "\n")

async def return_post(callback_url,answer):
    responseBody = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }
    if callback_url:
        async with httpx.AsyncClient() as client:
            response = await client.post(callback_url, json=responseBody)
            return response
        
async def process_and_post(request):
    # utterance 파싱
    utterance = request['userRequest']['utterance']
    # callbackUrl 파싱
    callback_url = request['userRequest']['callbackUrl']
    server_response = await get_mcp_server_response(utterance)
    news_doc = get_news_origin(server_response)
    # result_text = get_news_origin(server_response)
    answer = await general_llm(news_doc,utterance)
    result = await return_post(callback_url,answer)
    print("return_post", result)
    save_log(utterance,server_response,news_doc,answer)
    return



# 8. FastAPI 앱 생성
app = FastAPI()

@app.post("/basic")
async def kakao_question(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    background_tasks.add_task(process_and_post, data)

    return {
      "version": "2.0",
      "useCallback": True,
      "data": {
        "text": "생각하고 있는 중이에요😘 \n 30초 정도 소요될 거 같아요 기다려 주실래요?!"
      }
    }
