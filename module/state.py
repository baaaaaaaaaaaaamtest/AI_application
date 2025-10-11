from typing import Annotated, List, Literal, Tuple
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class State(TypedDict):
    question: Annotated[str, "user input question"]  # 사용자 질의 or requeustion 질의
    plan: Annotated[list[str], "get plan_node"]  # llm 생성한 작업 계획서
    messages: Annotated[list, add_messages]  # 작업 수행 후 얻은 데이터
    current_steps: Annotated[str, "current step"]  # 현재 단계에서 수행한 결과
    past_steps: Annotated[list, add_messages]  # 현재 단계에서 수행한 결과
    answer: Annotated[str, " output final answer"]  # 최종 답변 출력
    human_feedback: bool = False
    db_query: Annotated[str, "db_query"]  # 현재 단계에서 수행한 결과
    next_agent: Annotated[str, " use agent"]
    summary: Annotated[str, " 요약"]


class SubState(TypedDict):
    question: Annotated[str, "user input question"]  # 사용자 질의 or requeustion 질의
    plan: Annotated[list[str], "get plan_node"]  # llm 생성한 작업 계획서
    messages: Annotated[list, add_messages]  # 작업 수행 후 얻은 데이터
    current_steps: Annotated[str, "current step"]  # 현재 단계에서 수행한 결과
    past_steps: Annotated[list, add_messages]  # 현재 단계에서 수행한 결과
    answer: Annotated[str, " output final answer"]  # 최종 답변 출력
    human_feedback: bool = False
    db_query: Annotated[str, "db_query"]  # 현재 단계에서 수행한 결과
    next_agent: Annotated[str, " use agent"]
    summary: Annotated[str, " 요약"]

class DataState(TypedDict):
    question: Annotated[str, "user input question"]  # 사용자 질의 or requeustion 질의
    plan: Annotated[list[str], "get plan_node"]  # llm 생성한 작업 계획서
    messages: Annotated[list, add_messages]  # 작업 수행 후 얻은 데이터
    current_steps: Annotated[str, "current step"]  # 현재 단계에서 수행한 결과
    past_steps: Annotated[list, add_messages]  # 현재 단계에서 수행한 결과
    answer: Annotated[str, " output final answer"]  # 최종 답변 출력
    human_feedback: bool = False
    code: Annotated[str, "code"]  # 현재 단계에서 수행한 결과
    next_agent: Annotated[str, " use agent"]
    summary: Annotated[str, " 요약"]
