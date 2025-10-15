import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("utils"), "..")))
from module.utils import *
from module.prompt import *
from module.custom_model import *
from module.tools import *
from module.db_agent import *
from module.data_analysis_agent import *
from module.conversaction_agent import *
from module.tavily_agent import *
from typing_extensions import TypedDict

from typing import Annotated, List, Literal, Tuple
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage,
)
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph_supervisor import create_supervisor


class State(TypedDict):
    question: Annotated["str", "human request to llm"]
    summary: Annotated[str, "previous messages summarization "]
    messages: Annotated[list, add_messages]  # 화면 출력용
    answer: Annotated["str", "llm generate answer"]


def get_prompt_summary():
    template = """ 
        #System :
        As a summary-only node in LangGraph, you are responsible for summarizing key information by receiving records from previous conversations transcript.  
        - Focus on the core, essential points only. Remove any redundant information, emotional expressions, or unnecessary details.
        - Include all important numerical data or statistics mentioned in the conversation.
        - Keep the final 'Answer' from the last conversation turn intact in its original form. This is to preserve context for potential follow-up questions.
        - The summary should be clear, concise, and structured.

        # User Request :
        {question}
        
        # Conversation Transcript : 
        {messages}

        # Last Answer :
        {answer}

        # Previous Summary : 
        {summary}
        
        # Expected Output Format : 
        1. Key Summary:
        2. Important Numerical Data:
        3. Full Text of the Last Answer:
    """
    return ChatPromptTemplate.from_template(template)


def get_prompt_supervisor():
    template = """ 
        # System :
        You are a supervisor agent responsible for deciding which tool or method to use in response to a user request. 
        You have access to three specialized agents and the ability to answer directly without using tools.

        # Tool List: 
        1. db_agent:
        Purpose: Interact with a database containing sample music-related data.
        Capabilities: Query, insert, delete, and update records.
        Use this agent when the request involves finding information about music data, modifying data, or performing structured database operations.
        
        2. data_analysis_agent:
        Purpose: Perform data analysis using Python code.
        Capabilities: Generate and execute Python scripts for analysis, visualization, and statistics.
        Use this agent when the request requires data processing, analysis, or computation beyond simple retrieval.
        
        3. web_agent:
        Purpose: Use web search tools to solve problems by collecting external data.
        Features: Generating news, blogs, and information-based answers from external data retrieval
        Use this agent when you need data collection, retrieval,etc.., not everyday conversations, database, visualization, Python code generation.

        # User Input :
        {messages}

        # Direct Conversation:
        When the user’s request is casual or conversational and does not require data retrieval or analysis, 
        answer directly in Korean, without calling any tools.

        # Your responsibilities:
        Analyze the user’s input and determine whether to route the request to db_agent, data_analysis_agent, web_agent, or handle it directly.
        Always write in Korean in the final answer by adding various views, thoughts, trends, etc. that analyzed the previous data.
    """
    return ChatPromptTemplate.from_template(template)


def summary_node(state: State):
    """
    이전 모든 대화 내용중 핵심을 포함하여 요약하기위한 노드
    messages 가 6개 이상인 경우 summarization 수행
    """
    question = state.get("question", "")
    messages = state.get("messages", "")
    answer = state.get("answer", "")
    summary = state.get("summary", "")
    chain = get_prompt_summary() | get_gemini()
    response = chain.invoke(
        {
            "question": question,
            "messages": messages,
            "answer": answer,
            "summary": summary,
        }
    )
    if len(messages) > 2:
        # 오래된 메시지 삭제
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        # 요약 정보 반환
        return {
            "summary": response.content,
            "messages": delete_messages,
        }
    else:
        return {
            "summary": response.content,
        }


def supervisor_node(state: State):
    """
    두개의 에이전트를 자율적으로 선택하여 사용하는 관리자에이전트 구조
    """
    summary = state.get("summary", "")
    question = state["question"]
    content = f"Previous Conversations: {summary} \n Human Request : {question}"
    prompt = get_prompt_supervisor()

    db = get_db_agent()
    data_analysis = get_data_analysis_agent()
    web_agent = get_web_agent()
    supervisor = create_supervisor(
        model=get_gpt(),
        agents=[
            db,
            data_analysis,
            web_agent
        ],
        prompt=prompt,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()
    inputs = {"messages": [{"role": "user", "content": content}]}
    config = get_runnable_config(recursion_limit=15, thread_id=get_random_uuid())
    fianl_mode = ""
    final_data = ""
    for mode, data in supervisor.stream(
        inputs, config, stream_mode=["values", "messages"]
    ):
        fianl_mode = mode
        final_data = data
    return {"messages": final_data["messages"], "answer": final_data["messages"][-1]}


def get_graph():
    state_graph = StateGraph(State)
    state_graph.add_node("summary_node", summary_node)
    state_graph.add_node("supervisor_node", supervisor_node)

    state_graph.add_edge(START, "summary_node")
    state_graph.add_edge("summary_node", "supervisor_node")
    state_graph.add_edge("supervisor_node", END)

    cp = get_check_pointer()
    return state_graph.compile(checkpointer=cp)


def get_config():
    return get_runnable_config(recursion_limit=10, thread_id=get_random_uuid())
