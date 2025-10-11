import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("utils"), "..")))
from module.state import *
from module.prompt import *
from module.custom_model import *
from module.base_model import *
from module.tools import *
from module.db_query import *


def get_prompt_data_analysis():
    template = """ 
    You are an AI assistant
    It analyzes the user's questions and answers and previous conversation records to analyze, compare, and visualize data. 

    # User Request : 
    {current_steps}

    # Summary of conversation earlier: : 
    {summary}
    
    # placeholder:
    {messages}

    # Important :
    If there is no special request from the user, please make your own judgment and select the chart. 
    Make sure to answer in English when writing the code because errors may occur when writing in Korean.

    """
    return ChatPromptTemplate.from_template(template)


def get_prompt_answer():
    prompt = """ 
    You are a data analysis expert
    Print your final answer based on the python code you ran previously



    # User Request : 
    {current_steps}
    
    # Summary of conversation earlier :
    {summary}

    # placeholder:
    {messages}

    # important: 
    you must always answer in Korean.
    """
    return ChatPromptTemplate.from_template(prompt)


def get_python_tool():
    """
    create python code
    """
    return PythonAstREPLTool()


def instruction_node(state: DataState):
    plan = state.get("plan", [state["question"]])
    text_plan = "\n".join(f"{idx+1}. {text}" for idx, text in enumerate(plan))
    task = plan[0]
    task_str = f"""For the following plan: \n\n {text_plan} \n\n You are tasked with executing [step 1. {task}]."""
    return DataState({"messages": task_str, "current_steps": task_str})


def data_analysis_node(state: DataState):
    messages = state.get("messages", "")
    current_steps = state["current_steps"]
    summary = state.get("summary", "")
    python_tool = get_python_tool()
    tools = [python_tool]  # 필요 시 여러 도구 추가
    llm = get_gpt().bind_tools(tools, tool_choice="python_repl_ast")
    chain = get_prompt_data_analysis() | llm

    result = chain.invoke(
        {"current_steps": current_steps, "messages": messages, "summary": summary}
    )
    code = result.tool_calls[0]["args"]["query"]
    return DataState({"messages": code, "code": code})


def answer_node(state: DataState):
    prompt = get_prompt_answer()
    llm = get_gpt()
    query_gen_llm = prompt | llm
    summary = state.get("summary", "")
    current_steps = state["current_steps"]
    messages = state["messages"]
    result = query_gen_llm.invoke(
        {"summary": summary, "current_steps": current_steps, "messages": messages}
    )
    latest_messages = f" This step is done : {result.content}"
    code = state["code"]
    exec(code)
    return DataState(
        {
            "past_steps": result,
            "messages": latest_messages,
            "answer": latest_messages,
        }
    )


def get_data_analsys_graph():
    sub_state_graph = StateGraph(DataState)
    sub_state_graph.add_node("instruction_node", instruction_node)
    sub_state_graph.add_node("data_analysis_node", data_analysis_node)
    sub_state_graph.add_node("answer_node", answer_node)

    sub_state_graph.add_edge(START, "instruction_node")
    sub_state_graph.add_edge("instruction_node", "data_analysis_node")
    sub_state_graph.add_edge("data_analysis_node", "answer_node")
    sub_state_graph.add_edge("answer_node", END)

    sub_ck = get_check_pointer()
    sub_graph = sub_state_graph.compile(checkpointer=sub_ck)
    return sub_graph

