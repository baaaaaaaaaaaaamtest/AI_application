from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.document_loaders import WebBaseLoader
import json
import os
import bs4

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
        "display": 50,  # 최대 100
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
    if len(links_naver_news) != 0:
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
    