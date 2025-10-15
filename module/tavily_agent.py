import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("utils"), "..")))
from module.utils import *
from module.prompt import *
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from typing import Annotated, List
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from operator import itemgetter
from langchain_core.runnables import RunnableLambda


def get_prompt_web_search():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """ 
                # Instructions:
                You are a professional web search agent.
                It is important to carefully analyze and fully understand the requirements, which can lead to high-quality answers.
                Follow these steps strictly for optimal information retrieval and response generation:

                # Steps:
                1. If the requirements are complex or multifaceted, divide them into keyword groups and perform separate searches for each group.
                
                2. Use the keywords 'tavily_search_tool' for each group to gather information.

                3. Combine all the collected information and solve the problem

                4. Finally, it produces clear, concise, and complete answers in Korean.

                # Important:
                Follow this workflow sequentially: 
                Step 1 : (User Query Analysis and Segmentation) -> 
                Step 2 : (Tavily_search_tool) -> 
                Step 3 : (Merge Information Collected) -> 
                Step 4 : (final Korean answer)
                """,
            ),
            ("human", "{messages}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


def get_prompt():
    template = """당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
    You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.
    모든 대답은 반드시 한국말로 대답해주세요.
    
    # User Question :
    {question}
    
    # Context :
    {context}

    # Answer :
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt


def crawling(urls):
    docs = []
    if len(urls) > 0:
        web_loader = WebBaseLoader(
            web_paths=(urls),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer("body")),
        )
        results = web_loader.load()
        docs = results

    return docs


def get_split_docs(news_doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return text_splitter.split_documents(news_doc)


def get_retriever(split_docs, model="text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=model)
    db = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    return db.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.25, "fetch_k": 10}
    )


def get_bm25_retriever(split_docs):
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_retriever.k = 5  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.
    return bm25_retriever


def get_reranker(esenmble_retriever, user_request):
    docs = esenmble_retriever.invoke(user_request)
    documents_text = [doc.page_content for doc in docs]
    # print(documents_text)
    co = get_cohere_raranker()
    response = co.rerank(
        query=user_request,
        documents=documents_text,
        model="rerank-multilingual-v3.0",
        top_n=6,  # 상위 3개 결과만 리랭킹 반환
    )
    reranked_docs = []
    for result in response.results:
        reranked_docs.append(
            {
                # ** 연산자는 딕셔너리의 키-값 쌍을 풀어헤쳐 새로운 딕셔너리에 넣을 때 사용합니다.
                # 예를 들어, a = {'x':1, 'y':2}, b = {'z':3} 라면 {**a, **b}는 {'x':1, 'y':2, 'z':3}가 됩니다.
                **docs[result.index].dict(),
                "rerank_score": result.relevance_score,
            }
        )
    return reranked_docs


def reorder_documents(docs):
    # 재정렬
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    return reordered_docs


def get_esenmble_retriever(retriever1, retriever2):
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2],
        weights=[0.6, 0.4],  # 각 리트리버의 가중치를 설정합니다.
        k=6,  # 최종적으로 반환할 문서의 개수를 설정합니다.
    )
    return ensemble_retriever


# 웹검색시 키워드검색과 문장 검색 효율성 논문 찾아보기
@tool
def tavily_search_tool(
    user_request: Annotated[str, "user_request sentence"],
) -> list[str]:
    """
    사용자의 요청을 분석하여 실제 검색을 수행하기위한 도구

    Args:
    user_request(str) : 사용자의 요구사항

    Return :
    any
    """
    tool = TavilySearch(
        max_results=5,
        topic="general",
        # include_domains=["kr.wikipedia.org", "news.naver.com","entertain.naver.com","sports.naver.com"],
        exclude_domains=["youtube.com", "youtubekids.com"],
    )
    llm = get_gpt()
    prompt = get_prompt()
    response = tool.invoke({"query": user_request})
    urls = [result["url"] for result in response["results"]]
    docs = crawling(urls)
    split_docs = get_split_docs(docs)
    faiss_retriever = get_retriever(split_docs)
    bm25_retriever = get_bm25_retriever(split_docs)
    esenmble_retriever = get_esenmble_retriever(faiss_retriever, bm25_retriever)
    reranker_docs = get_reranker(esenmble_retriever, user_request)
    reorder_docs = reorder_documents(reranker_docs)
    chain = prompt | llm
    return chain.invoke({"question": user_request, "context": reorder_docs})


def get_web_agent():
    prompt = get_prompt_web_search()
    llm = get_gemini()
    return create_react_agent(
        prompt=prompt, model=llm, tools=[tavily_search_tool], name="web_search"
    )
