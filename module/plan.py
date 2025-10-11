import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("utils"), "..")))
from module.state import *
from module.prompt import *
from module.custom_model import *
from module.base_model import *
from module.tools import *
from module.db_query import *


def plan_node(state: State) -> State:
    llm = get_gpt()
    prompt = get_prompt_music_planner()
    chain = prompt | llm.with_structured_output(PlanListModel)
    response = chain.invoke({"messages": [state["question"]]})
    return State({"plan": response.steps, "messages": response.steps})


def next_agent_node(state: State) -> State:
    plan = state.get("plan", [state["question"]])
    next_task = plan[0]
    prompt = get_prompt_routing_node()
    llm = get_gpt()
    chain = prompt | llm.with_structured_output(RouteModel)
    result = chain.invoke({"next_task": next_task})
    response = f"실행 계획 : {next_task} \n 사용 도구 : {result.datasource}"
    return State(
        {
            "current_steps": next_task,
            "messages": AIMessage(content=response),
            "next_agent": result.datasource,
        }
    )


def final_node(state: State) -> State:
    llm = get_gemini()
    chain = get_prompt_final() | llm
    result = chain.invoke(
        {
            "current_steps": state["current_steps"],
            "next_agent": state["next_agent"],
            "plan": state.get("plan"),
        }
    )
    return State({"messages": [result], "answer": result.content})


def get_plan_graph():
    state_graph = StateGraph(State)
    state_graph.add_node("plan_node", plan_node)
    state_graph.add_node("next_agent_node", next_agent_node)
    state_graph.add_node("final_node", final_node)

    state_graph.add_edge(START, "plan_node")
    state_graph.add_edge("plan_node", "next_agent_node")
    state_graph.add_edge("next_agent_node", "final_node")
    state_graph.add_edge("final_node", END)

    cp = get_check_pointer()
    return state_graph.compile(checkpointer=cp)
