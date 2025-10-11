import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("utils"), "..")))

from typing import Literal
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

from module.utils import *
from module.prompt import *
from module.state import *
from langgraph.graph import StateGraph, START, END
from module.custom_model import *


# @tool
# def db_query_tool(query: str) -> str:
#     """
#     Run SQL queries against a database and return results
#     Returns an error message if the query is incorrect
#     If an error is returned, rewrite the query, check, and retry
#     """
#     # 쿼리 실행
#     db = get_db()
#     result = db.run_no_throw(query)

#     # 오류: 결과가 없으면 오류 메시지 반환
#     if "Error" in result:
#         return f"Error: {result} \n\n . Please rewrite your query and try again."
#     # 정상: 쿼리 실행 결과 반환
#     elif not result:
#         return "Success: value is None"
#     else:
#         return f"Success: {result}"


# def instruction_node(state: SubState):
#     plan = state.get("plan", [state["question"]])
#     text_plan = "\n".join(f"{idx+1}. {text}" for idx, text in enumerate(plan))
#     task = plan[0]
#     task_str = f"""For the following plan: \n\n {text_plan} \n\n You are tasked with executing [step 1. {task}]."""
#     return SubState({"messages": task_str, "current_steps": task_str})


# def get_table_list_node(state: SubState):
#     llm = get_gpt()
#     tools = get_db_tool(llm)
#     sql_db_list_tables = next(
#         tool for tool in tools if tool.name == "sql_db_list_tables"
#     )
#     llm_get_schema = llm.bind_tools(
#         [sql_db_list_tables], tool_choice="sql_db_list_tables"
#     )
#     return SubState({"messages": llm_get_schema.invoke("")})


# def get_all_table_node(state: SubState):
#     llm = get_gpt()
#     tools = get_db_tool(llm)
#     sql_db_list_tables = next(
#         tool for tool in tools if tool.name == "sql_db_list_tables"
#     )
#     return ToolNode([sql_db_list_tables])


# def get_one_table_info_node(state: SubState):
#     llm = get_gpt()
#     tools = get_db_tool(llm)
#     sql_db_schema = next(tool for tool in tools if tool.name == "sql_db_schema")
#     llm_with_schema = llm.bind_tools([sql_db_schema], tool_choice="sql_db_schema")
#     state_message = state["messages"][-3:]
#     result = llm_with_schema.invoke(state_message)
#     return SubState({"messages": result})


# def get_one_table_schema_node(state: SubState):
#     llm = get_gemini()
#     tools = get_db_tool(llm)
#     sql_db_schema = next(tool for tool in tools if tool.name == "sql_db_schema")
#     return ToolNode([sql_db_schema])


# def get_query_gen_node(state: SubState):
#     prompt = get_prompt_query_gen()
#     llm = get_gpt()
#     query_gen_llm = prompt | llm
#     history = state["messages"]
#     current_steps = state["current_steps"]
#     query_gen = query_gen_llm.invoke(
#         {"placeholder": history, "current_steps": current_steps}
#     )
#     sql_query = query_gen.content
#     return SubState({"messages": query_gen, "db_query": sql_query})


# def check_query_relavant(state: SubState):
#     prompt = get_prompt_relevant_query()
#     llm = get_gpt()
#     query_gen_llm = prompt | llm.with_structured_output(GradeQuery)
#     current_steps = state["current_steps"]
#     query = state["db_query"]
#     query_gen = query_gen_llm.invoke({"current_steps": current_steps, "query": query})
#     return SubState({"messages": query_gen.datasource})


# def execute_query(state: SubState):
#     query = state["db_query"]
#     response = db_query_tool(query)
#     return SubState({"messages": AIMessage(content=response)})


# #  실패시 get_query_check_node -> execute_query 로직 추가
# def get_query_check_node(state: SubState):
#     prompt = get_prompt_query_check()
#     llm = get_gpt().bind_tools([db_query_tool], tool_choice="db_query_tool")
#     chain = prompt | llm
#     history = state["messages"][-1].tool_calls[0]["args"]["query"]
#     query_gen = chain.invoke({"placeholder": [history]})
#     return SubState({"messages": query_gen})


# def answer_node(state: SubState):
#     prompt = get_prompt_answer()
#     llm = get_gpt()
#     query_gen_llm = prompt | llm
#     messages = state["messages"]
#     current_steps = state["current_steps"]
#     summary = state.get("summary", "")
#     result = query_gen_llm.invoke(
#         {"messages": messages, "current_steps": current_steps, "summary": summary}
#     )
#     latest_messages = f" This step is done : {result.content}"
#     return SubState(
#         {
#             "past_steps": result,
#             "messages": latest_messages,
#             "answer": latest_messages,
#         }
#     )


# def query_relevant(
#     state: SubState,
# ) -> Literal["execute_query", "get_one_table_info_node"]:
#     latest_messages: str = state["messages"][-1].content
#     if latest_messages == "yes":
#         return "execute_query"
#     else:
#         return "get_one_table_info_node"


# def get_db_graph():
#     sub_state_graph = StateGraph(SubState)
#     sub_state_graph.add_node("instruction_node", instruction_node)
#     sub_state_graph.add_node("get_table_list_node", get_table_list_node)
#     sub_state_graph.add_node("get_all_table_node", get_all_table_node)
#     sub_state_graph.add_node("get_one_table_info_node", get_one_table_info_node)
#     sub_state_graph.add_node("get_one_table_schema_node", get_one_table_schema_node)
#     sub_state_graph.add_node("get_query_gen_node", get_query_gen_node)
#     sub_state_graph.add_node("check_query_relavant", check_query_relavant)

#     sub_state_graph.add_node("get_query_check_node", get_query_check_node)
#     sub_state_graph.add_node("execute_query", execute_query)
#     sub_state_graph.add_node("answer_node", answer_node)

#     sub_state_graph.add_edge(START, "instruction_node")
#     sub_state_graph.add_edge("instruction_node", "get_table_list_node")
#     sub_state_graph.add_edge("get_table_list_node", "get_all_table_node")
#     sub_state_graph.add_edge("get_all_table_node", "get_one_table_info_node")
#     sub_state_graph.add_edge("get_one_table_info_node", "get_one_table_schema_node")
#     sub_state_graph.add_edge("get_one_table_schema_node", "get_query_gen_node")
#     sub_state_graph.add_edge("get_query_gen_node", "check_query_relavant")
#     sub_state_graph.add_conditional_edges(
#         source="check_query_relavant", path=query_relevant
#     )

#     sub_state_graph.add_edge("execute_query", "answer_node")
#     sub_state_graph.add_edge("answer_node", END)

#     sub_ck = get_check_pointer()
#     sub_graph = sub_state_graph.compile(checkpointer=sub_ck)
#     return sub_graph
