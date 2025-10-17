import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("utils"), "..")))
from module.utils import *
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from typing import Annotated


def get_prompt_data_analysis():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """ 
                You are a helpful assistant.
                You are an intelligent assistant that analyzes the user's request, writes Python code that fulfills the need,and uses the python_repl tool to execute and visualize the code results.
                # Instructions :
                - Analyze user's input carefully.
                - Write valid Python code for the task.
                - After writing the code, run it using the python_repl tool.
                - Provide the final output after executing the code.
                - If return 'Success', go to '__end__'.
                - Please provide various views on the analysis before the final output.

                # Important :
                - Be sure to execute 'run_python_repl' only once for code made of 'generate_python_code'.
                - You must final answer of korean
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{messages}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


def get_prompt_python_code():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """ 
                # Instructions:
                - It is an AI tool that analyzes users and writes them in Python code
                - To create Python code for chart generation, you need to analyze your meaning accurately.
                - When you generate Python code, you must write all text in English, including the title, description, and variables.
                - The output must be written in Python code
               
                # Important:
                  You can not generate plt.show().
                  Therefore, the analysis results can be saved as files.
  
                # For example:
                    - step1 : file_path = 'current_time.png'
                    - step2 : plt.show() -> plt.savefig ('file_path.png', dpi=300)
                    - step3 : print('file_path.png')

                """,
            ),
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

    Args:
    user_input (string): 사용자 요구사항

    """

    llm = get_gemini()
    chain = get_prompt_python_code() | llm
    return chain.invoke({"messages": user_input})


@tool
def run_python_repl(
    python_code: Annotated[str, "The python code to execute to generate your chart."],
):
    """
    # System :
    You are a tool to execute Python code.
    The following instructions shall be observed.

    # Instructions :
    1. Never run the `plt.show()` code.
    2. Make sure to save it as a file when creating a chart.
    3. Make sure to print out the saved path and file name.
    4. You must return fileName

    Args:
    python_code (string): 사용자 요구를 기반으로 생성한 파이썬 코드

    Returns:
    str : filepath or etc..
    """
    try:
        # 주어진 코드를 Python REPL에서 실행하고 결과 반환
        python_repl = get_python_repl()
        result = python_repl.run(python_code)
        return "Success: " + result
    except BaseException as e:
        return f"Error: {repr(e)}"
    # 실행 성공 시 결과와 함께 성공 메시지 반환


def get_data_analysis_agent():
    # python_repl = get_python_repl()
    # 3. LLM 및 프롬프트 생성
    llm = get_gpt()  # 실제 LLM 인스턴스 또는 래퍼 전달
    prompt = get_prompt_data_analysis()
    # 4. 에이전트 생성
    return create_react_agent(
        model=llm,
        tools=[generate_python_code, run_python_repl],
        prompt=prompt,
        name="data_analysis_agent",
        # debug=True,
    )
