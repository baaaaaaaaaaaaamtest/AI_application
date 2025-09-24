from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.document_loaders import WebBaseLoader
import json
import os


def get_mcp_client():
    return MultiServerMCPClient(
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
            },
                "gmail-mcp": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@shinzo-labs/gmail-mcp",
                    "--key",
                    "2ca5a27c-7a2f-49e7-b2fe-b48148920235",
                    "--profile",
                    "sad-vulture-QiAFDw"
                ],
                "transport": "stdio",
            }
        }
    )