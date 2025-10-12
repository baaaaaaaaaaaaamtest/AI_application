import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("utils"), "..")))
from module.state import *
from module.prompt import *
from module.custom_model import *
from module.utils import *
from module.tools import *
from langgraph.prebuilt import create_react_agent


def get_prompt_data_analysis():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """ 
                You are a helpful assistant.
                You are an intelligent assistant that analyzes the user's request, writes Python code that fulfills the need,and uses the python_repl tool to execute and visualize the code results.
                Instructions #
                - Analyze user's input carefully.
                - Write valid Python code for the task.
                - After writing the code, run it using the python_repl tool.
                - Provide the final output after executing the code.
                
                Important #
                - Be sure to execute 'python_repl' only once for code made of 'generate_python_code'.
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{messages}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


@tool
def generate_python_code(user_input):
    """
    You are an AI tool that analyzes users and writes them in Python code
    You must accurately analyze the meaning of the user to create a Python code for chart generation.
    When generating Python code, you must write all text in English, including the title, description, and variables.
    Make sure you write the output in python code
    """

    llm = get_gemini()
    return llm.invoke(user_input)


def get_data_analysis_agent():
    python_repl = get_python_repl()
    # 3. LLM 및 프롬프트 생성
    llm = get_gpt()  # 실제 LLM 인스턴스 또는 래퍼 전달
    prompt = get_prompt_data_analysis()
    # 4. 에이전트 생성
    return create_react_agent(
        model=llm,
        tools=[generate_python_code, python_repl],
        prompt=prompt,
        name="data_analysis_agent",
    )
