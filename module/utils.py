
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_core.tools.base import BaseTool,Field  # BaseTool이 있는 경로에 따라 import 수정
import os
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Literal, Union,Optional
from langchain_core.messages import AnyMessage
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_graph, invoke_graph, random_uuid
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_teddynote import logging
from langchain_core.retrievers import BaseRetriever
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
load_dotenv()

# 프로젝트 이름을 입력합니다.
def strt_langsmith(name:str = "my_test_1"):
    """
        추적 가능하도록 lang_smith 활성화 
        
        Args:
            name : lang_smith project name
    """
    logging.langsmith(name)


def get_gpt(model:str = "gpt-4.1-mini",temperature:int=0):
    return ChatOpenAI(model=model,temperature=temperature)
# 모델 로드 
def get_gemini( 
        model:str = "gemini-2.5-flash-lite",
        temperature:int=0,
        max_output_tokens:int = 1096):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        google_api_key=os.getenv('GOOGLE_API_KEY'),
    )

def get_pdf_loader(path:str = "../data/SPRI_AI_Brief_2023년12월호_F.pdf"):
    """
        pdf loader 를 활용하여 pdf 파일을 Load함
        추후 docling,  한글 pdf loader 등 활용하여 성능 비교 필요함
    """
    return PyPDFLoader(path)

def get_text_splitter(chunk_size:int = 1000, chunk_overlap:int = 50):
    """
        ["\n\n", "\n", " ", ""] 를 기본적으로 사용하여 text를 자른다
        만약 "\n\n"을 사용하여 자른 chunk가 1000을 넘어가는경우 "\n" 사용하여 다시 자르는 행위를
        재귀적으로 수행하는 매커니즘
    """
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def get_docs(loader:PyPDFLoader,get_text_splitter)-> list[Document]:
    """PDF Loader 활용, 이미 정해진 splitter 활용하여 pdf 를 분할, doc로 생성함"""
    return loader.load_and_split(get_text_splitter)

def get_embedding(model:str = "models/gemini-embedding-001"):
    """
        텍스트 등 파일을 지정된 모델의 차원에 따라 숫자로 변환하는 기법
    """
    return GoogleGenerativeAIEmbeddings(model=model)


def get_retriever(docs:Document,embedding:Embeddings,k:int = 3)->BaseRetriever:
    """
        사전 만들어진 Document List 를 Embedding 활용하여 벡터화 진행
        벡터화 된 데이터를 FAISS, Chroma, Pinecone 등에 저장 가능함
        이후 검색 가능하드로고 retriever 제공함
    """
    vector = FAISS.from_documents(documents=docs, embedding=embedding)
    return vector.as_retriever(k=k)

def get_retriever_tool(retriever):
    return create_retriever_tool(
        retriever,
        name="pdf_search",  # 도구의 이름을 입력합니다.
        description="use this tool to search information from the PDF document",  # 도구에 대한 설명을 자세히 기입해야 합니다!!
        document_prompt=PromptTemplate.from_template(
        "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
        ),
    )

def get_tavily_tool():
    """ search 등 활용하여 직접 검색도 가능함 """
    return TavilySearch()

def get_tool_node(*tools):
    if len(tools) == 0 :
        raise TypeError(f"tools is empty")
    for tool in tools:
        if not isinstance(tool, BaseTool):
            raise TypeError(f"tool {tool} is not a BaseTool instance")
    return ToolNode(tools)

def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
    """Conditional routing function for tool-calling workflows.

    from langgraph.prebuilt import tools_condition 참조
   

    Example:
        Basic usage in a ReAct agent:

        ```python
        from langgraph.graph import StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition
        from typing_extensions import TypedDict

        class State(TypedDict):
            messages: list

        graph = StateGraph(State)
        graph.add_node("llm", call_model)
        graph.add_node("tools", ToolNode([my_tool]))
        graph.add_conditional_edges(
            "llm",
            tools_condition,  # Routes to "tools" or "__end__"
            {"tools": "tools", "__end__": "__end__"}
        )
        ```

        Custom messages key:

        ```python
        def custom_condition(state):
            return tools_condition(state, messages_key="chat_history")
        ```

    Note:
        This function is designed to work seamlessly with ToolNode and standard
        LangGraph patterns. It expects the last message to be an AIMessage when
        tool calls are present, which is the standard output format for tool-calling
        language models.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"

def get_check_pointer():
    return MemorySaver()

def get_runnable_config(recursion_limit:int = 10, thread_id:str=random_uuid()):
    print(thread_id)
    return RunnableConfig(recursion_limit=recursion_limit, configurable={"thread_id": thread_id})


from langchain_teddynote.evaluator import GroundednessChecker

def get_relevant(llm,target:str="question-retrieval"):
    return GroundednessChecker(llm=llm, target=target).create()

def convert_docs_str(docs):
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{doc.metadata['source']}</source><page>{int(doc.metadata['page'])+1}</page></document>"
            for doc in docs
        ]
    )


def convert_search_str(docs):
    return "\n".join(
        [
            f"<document><content>{doc['content']}</content><source>{doc['url']}</source></document>"
            for doc in docs
        ]
    )




def visualize_graph(graph, xray=False, ascii=False):
    from IPython.display import Image, display
    from langgraph.graph.state import CompiledStateGraph
    from dataclasses import dataclass
    """
    CompiledStateGraph 객체를 시각화하여 표시합니다.

    이 함수는 주어진 그래프 객체가 CompiledStateGraph 인스턴스인 경우
    해당 그래프를 Mermaid 형식의 PNG 이미지로 변환하여 표시합니다.

    Args:
        graph: 시각화할 그래프 객체. CompiledStateGraph 인스턴스여야 합니다.
        xray: 그래프 내부 상태를 표시할지 여부.
        ascii: ASCII 형식으로 그래프를 표시할지 여부.
    """ 
    
    @dataclass
    class NodeStyles:
        default: str = (
            "fill:#45C4B0, fill-opacity:0.3, color:#23260F, stroke:#45C4B0, stroke-width:1px, font-weight:bold, line-height:1.2"  # 기본 색상
        )
        first: str = (
            "fill:#45C4B0, fill-opacity:0.1, color:#23260F, stroke:#45C4B0, stroke-width:1px, font-weight:normal, font-style:italic, stroke-dasharray:2,2"  # 점선 테두리
        )
        last: str = (
            "fill:#45C4B0, fill-opacity:1, color:#000000, stroke:#45C4B0, stroke-width:1px, font-weight:normal, font-style:italic, stroke-dasharray:2,2"  # 점선 테두리
        )
    if not ascii:
        try:
            # 그래프 시각화
            if isinstance(graph, CompiledStateGraph):
                display(
                    Image(
                        graph.get_graph(xray=xray).draw_mermaid_png(
                            background_color="white",
                            node_colors=NodeStyles(),
                        )
                    )
                )
        except Exception as e:
            print(f"그래프 시각화 실패 (추가 종속성 필요): {e}")
            print("ASCII로 그래프 표시:")
            try:
                print(graph.get_graph(xray=xray).draw_ascii())
            except Exception as ascii_error:
                print(f"ASCII 표시도 실패: {ascii_error}")
    else:
        print(graph.get_graph(xray=xray).draw_ascii())

# Query Rewrite 프롬프트 정의
re_write_prompt = PromptTemplate(
    template="""Reformulate the given question to enhance its effectiveness for vectorstore retrieval.
- Analyze the initial question to identify areas for improvement such as specificity, clarity, and relevance.
- Consider the context and potential keywords that would optimize retrieval.
- Maintain the intent of the original question while enhancing its structure and vocabulary.

# Steps

1. **Understand the Original Question**: Identify the core intent and any keywords.
2. **Enhance Clarity**: Simplify language and ensure the question is direct and to the point.
3. **Optimize for Retrieval**: Add or rearrange keywords for better alignment with vectorstore indexing.
4. **Review**: Ensure the improved question accurately reflects the original intent and is free of ambiguity.

# Output Format

- Provide a single, improved question.
- Do not include any introductory or explanatory text; only the reformulated question.

# Examples

**Input**: 
"What are the benefits of using renewable energy sources over fossil fuels?"

**Output**: 
"How do renewable energy sources compare to fossil fuels in terms of benefits?"

**Input**: 
"How does climate change impact polar bear populations?"

**Output**: 
"What effects does climate change have on polar bear populations?"

# Notes

- Ensure the improved question is concise and contextually relevant.
- Avoid altering the fundamental intent or meaning of the original question.


[REMEMBER] Re-written question should be in the same language as the original question.

# Here is the original question that needs to be rewritten:
{question}
""",
    input_variables=["question"],
)