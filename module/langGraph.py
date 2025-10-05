from typing import TypedDict, Annotated, List, Literal,Tuple
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage,ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
import os,sys
sys.path.append(os.path.abspath('/home/ansgyqja/AI_application'))
from module.utils import * 
from module.prompt import * 
from module.custom_model import *
from module.base_model import *
from module.tools import *

########## 1. 상태 정의 ##########
class State(TypedDict):
    # 메시지 목록 주석 추가
    messages: Annotated[list, add_messages]
    ask_human: bool = False



@tool
def num_sum(a : int, b: int ):
    """ a and b total sum """
    return a+b

@tool
def num_division(a : int, b: int ):
    """ a and b division num """
    return a/b

@tool
def num_multiplication(a : int, b: int ):
    """ a and b  multiplication num"""
    return a*b

tavily = get_tavily_tool()
tools = [num_sum,num_division,num_multiplication,tavily]

template = """ 
    # System :
    You are an AI assistant
    Depending on your needs, you can use a number of tools or solve problems yourself.
    If you decide to use the tool or solve the problem yourself, make sure to do so after getting confirmation from the user

    # History : 
    {messages}

    # Important : 
    ask_human value : `{ask_human}`
    
    Be sure to follow the guide below according to the value of the ask_human!!

    1. If the `ask_human` value is True, answer the question without asking the user again for comment

    2. If the value 'ask_human' is false, ask for yes or no confirmation to use tool_calls

    Write your response in Korean.

"""

prompt = ChatPromptTemplate.from_template(
        template
    )


_template = """ 
    # System :
    You are an AI assistant
    Depending on your needs, you can use a number of tools or solve problems yourself.
    If you decide to use the tool or solve the problem yourself, make sure to do so after getting confirmation from the user

    # History : 
    {messages}

    # Impotant : 
    If there is one history, return 'no'
    If you have 2 or more history, analyze the last 2 messages and the AI asks a question and the human responds 'yes' and otherwise 'no'
    """
_prompt = ChatPromptTemplate.from_template(
        _template
    )


class HumanRequest(BaseModel):
    """ 
        사용자의 Question 의 내용을 분석하여 사용자의 순수한 질의, 요청일 경우 False 처리
        AI 가 어떠한 판단한 내용의 대한 사람의 피드백, 또는 응답일경우네느 True 로 처리

    """
    datasource : Literal['yes','no'] = Field(
        description = """
            If there is one history, return 'no'
            If you have 2 or more history, analyze the last 2 messages and the AI asks a question and the human responds 'yes' and otherwise 'no'
            """
    )
def define_node(state:State):
    llm = get_gemini()
    chain =  _prompt| llm.with_structured_output(HumanRequest)
    result = chain.invoke({'messages':state["messages"]})
    if result.datasource == 'yes':
        ask_human = True
    else : 
        ask_human = False
    print(result)
    return State({"ask_human":ask_human})


def chatbot(state:State):
    llm = get_gemini()
    llm_with_tools = llm.bind_tools(tools)
    chain = prompt | llm_with_tools
    result = chain.invoke({"messages":state["messages"],"ask_human":state['ask_human']})
    return State({"messages": [result]})

tool_node = ToolNode(tools)

def final_node(state:State):
    llm = get_gemini()
    llm_with_tools = llm.bind_tools(tools)
    chain = prompt | llm_with_tools
    result = chain.invoke({"messages":state["messages"],"ask_human":state['ask_human']})
    State({"messages": [result]})
    
def should_continue(state:State)->Literal['tools','final_node']:
    ask_human = state['ask_human']
    if ask_human:
        return 'tools'
    else:
        return 'final_node'
    
    
state_graph = StateGraph(State)
state_graph.add_node('define_node',define_node)
state_graph.add_node('chatbot',chatbot)
state_graph.add_node('tools',tool_node)
state_graph.add_node('final_node',final_node)

state_graph.add_edge(START,'define_node')
state_graph.add_edge('define_node','chatbot')

state_graph.add_conditional_edges(
    source='chatbot',
    path=should_continue
)

state_graph.add_edge('tools','final_node')
state_graph.add_edge('final_node',END)
ck = get_check_pointer()
graph = state_graph.compile(checkpointer=ck)



config = get_runnable_config(recursion_limit=10,thread_id=get_random_uuid())
# user='오늘 날씨에 듣기좋은 노래 추천해줘'
# inputs = {"messages": [HumanMessage(content=user)]}
# for event in graph.stream(config=config,input=inputs,stream_mode='values'):    
#     print(event)

def get_config():
    return config
def get_graph():
    return graph