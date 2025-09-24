from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
from typing import List
import json

import bs4


def get_news_origin(data)->List[Document]:
    json_data = json.loads(data)
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


async def invoke_messages(tool, messages):
    for msg in messages:
        msg_id = msg['id']
        result = await tool.ainvoke({'id':msg_id})
        data = json.loads(result)  # JSON 문자열을 dict로 변환
        print(f"Response for {msg_id}: {data['snippet']}")
    

    