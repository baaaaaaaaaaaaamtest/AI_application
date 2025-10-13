import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
import os, sys

sys.path.append(os.path.abspath("C:\\Users\\ansgy\\IdeaProjects\\AI_application"))
# from module.handler_copy import stream_handler, format_search_result
from module.langGraph import *

st.set_page_config(page_title="K ChatGPT 💬", page_icon="💬")
st.title("K ChatGPT 💬")

start_langsmith("streamlit_01_practice")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def get_double_array(messages):
    result = []
    temp_list = []
    previous_role = None
    for msg_with_type in messages:

        role = msg_with_type.chat_message.role
        if role == previous_role:
            # 같은 role이면 현재 그룹에 추가
            temp_list.append(msg_with_type)
        else:
            # role이 바뀌면 이전 그룹을 결과에 추가하고 새 그룹 시작
            if temp_list:
                result.append({previous_role: temp_list})
            temp_list = [msg_with_type]
            previous_role = role

    # 마지막 그룹 추가
    if temp_list:
        result.append({previous_role: temp_list})
    return result


def print_history():
    st.session_state["messages"] = [
        ChatMessageWithType(
            chat_message=ChatMessage(
                content="서울의 1월, 2월, 3월 평균 기온은 각각 약 -3.5°C, -0.8°C, 4.2°C 시각화해줘",
                additional_kwargs={},
                response_metadata={},
                role="user",
            ),
            msg_type="text",
            tool_name="",
        ),
        ChatMessageWithType(
            chat_message=ChatMessage(
                content="content=\"```python\\nimport matplotlib.pyplot as plt\\n\\n# Data for Seoul's average temperature in January, February, and March\\nmonths = ['January', 'February', 'March']\\naverage_temperatures = [-3.5, -0.8, 4.2]\\n\\n# Create the bar chart\\nplt.figure(figsize=(8, 6))\\nplt.bar(months, average_temperatures, color=['skyblue', 'lightcoral', 'lightgreen'])\\n\\n# Add titles and labels\\nplt.title('Average Temperature in Seoul (January-March)')\\nplt.xlabel('Month')\\nplt.ylabel('Average Temperature (°C)')\\nplt.grid(axis='y', linestyle='--', alpha=0.7)\\n\\n# Save the plot to a file\\nplt.savefig('seoul_average_temperature.png', dpi=300)\\n```\" additional_kwargs={} response_metadata={'safety_ratings': [], 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash-lite'} id='run--e8c88198-b303-4d9e-8a22-aaaf4e3ac449' usage_metadata={'input_tokens': 207, 'output_tokens': 184, 'total_tokens': 391, 'input_token_details': {'cache_read': 0}}",
                additional_kwargs={},
                response_metadata={},
                role="assistant",
            ),
            msg_type="tool_result",
            tool_name="generate_python_code",
        ),
        ChatMessageWithType(
            chat_message=ChatMessage(
                content="Success: seoul_average_temperature.png\n",
                additional_kwargs={},
                response_metadata={},
                role="assistant",
            ),
            msg_type="tool_result",
            tool_name="run_python_repl",
        ),
        ChatMessageWithType(
            chat_message=ChatMessage(
                content='서울의 1월, 2월, 3 월 평균 기온을 시각화한 막대 그래프를 생성했습니다. 그래프는 각 월별 평균 기온을 색상으로 구분하여 보여주며, y축에는 온도(°C)가 표시되어 있습니다. 생성된 그래프 파일명은 "seoul_average_temperature.png"입니다. 필요하시면 이 파일을 확인해 주세요.',
                additional_kwargs={},
                response_metadata={},
                role="assistant",
            ),
            msg_type="text",
            tool_name="",
        ),
    ]
    # print(f'print history : \n {st.session_state["messages"]}\n\n')

    messages = get_double_array(st.session_state["messages"])

    for msg in messages:
        # st.chat_message(msg.chat_message.role).write(msg.chat_message.content)
        role = list(msg.keys())[0]
        print(role)
        if role == "user":
            for m in msg[role]:
                user_role = m.chat_message.role
                user_content = m.chat_message.content
                user_tool_name = m.tool_name
                user_msg_type = m.msg_type
                st.chat_message(user_role).write(user_content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                for m in msg[role]:
                    ai_role = m.chat_message.role
                    ai_content = m.chat_message.content
                    ai_tool_name = m.tool_name
                    ai_msg_type = m.msg_type
                    if ai_msg_type == "text":
                        st.markdown(ai_content)
                    elif ai_msg_type == "tool_result":
                        with st.expander(f"✅ {ai_tool_name}"):
                            if ai_tool_name == "run_python_repl":
                                print(f"metadata : {ai_content} \n")
                                filename = ai_content.replace("Success: ", "").strip()
                                st.image(filename)
                            st.markdown(ai_content)
        #     pass
        # role = msg.chat_message.role
        # content = msg.chat_message.content
        # tool_name = msg.tool_name
        # msg_type = msg.msg_type
        # if role == "user":
        #     st.chat_message(role).write(content)
        # else:
        #     with st.chat_message("assistant"):
        #         if msg_type == "text":
        #             st.markdown(content)
        #         elif msg_type == "tool_result":
        #             with st.expander(f"✅ {tool_name}"):
        #                 if tool_name == "run_python_repl":
        #                     print(f"metadata : {content} \n")
        #                     filename = content.replace("Success: ", "").strip()
        #                     st.image(filename)
        #                 st.markdown(content)

        # print(f"tool_name: {tool_name}")
        # print(f"msg_type: {msg_type}")
        # print(f"content: {content}\n")

        # st.chat_message(msg.chat_message.role).write(msg.chat_message.content)


# def add_history(role, content):
#     st.session_state["messages"].append(ChatMessage(role=role, content=content))


with st.sidebar:
    clear_btn = st.button("대화내용 초기화")


if clear_btn:
    retriever = st.session_state["messages"].clear()
    st.session_state["chain"] = get_graph()
    st.session_state["config"] = get_config()


if "chain" not in st.session_state:
    st.session_state["chain"] = get_graph()
    st.session_state["config"] = get_config()


# if user_input := st.chat_input():
#     add_history("user", user_input)
#     st.chat_message("user").write(user_input)
#     with st.chat_message("assistant"):
#         chat_container = st.empty()
#         ai_answer = ""
#         inputs = {"question": user_input}
#         for step, metadata in st.session_state["chain"].stream(
#             input=inputs,
#             config=st.session_state["config"],
#             stream_mode="messages",
#             subgraphs=True,
#         ):
#             if step != ():
#                 ai_answer += metadata[0].content
#                 chat_container.markdown(ai_answer)
#         add_history("ai", ai_answer)


from attr import dataclass


def format_search_result(results):
    """
    Format search results into a markdown string.

    Args:
        results (str): JSON string containing search results

    Returns:
        str: Formatted markdown string with search results
    """
    # import json

    # results = json.loads(results)

    # answer = ""
    # for result in results:
    #     answer += f'**[{result["title"]}]({result["url"]})**\n\n'
    #     answer += f'{result["content"]}\n\n'
    #     answer += f'신뢰도: {result["score"]}\n\n'
    #     answer += "\n-----\n"
    return results


def get_current_tool_message(tool_args, tool_call_id):
    """
    Get the tool message corresponding to the given tool call ID.

    Args:
        tool_args (list): List of tool arguments
        tool_call_id (str): ID of the tool call to find

    Returns:
        dict: Tool message if found, None otherwise
    """
    if tool_call_id:
        for tool_arg in tool_args:
            if tool_arg["tool_call_id"] == tool_call_id:
                return tool_arg
        return None
    else:
        return None


def stream_handler(streamlit_container, agent_executor, inputs, config):
    """
    Handle streaming of agent execution results in a Streamlit container.

    Args:
        streamlit_container (streamlit.container): Streamlit container to display results
        agent_executor: Agent executor instance
        inputs: Input data for the agent
        config: Configuration settings

    Returns:
        tuple: (container, tool_args, agent_answer)
            - container: Streamlit container with displayed results
            - tool_args: List of tool arguments used
            - agent_answer: Final answer from the agent
    """
    # Initialize result storage
    tool_args = []
    agent_answer = ""
    agent_message = None  # Pre-declare agent_message variable

    container = streamlit_container.container()
    with container:
        for chunk_msg, metadata in agent_executor.stream(
            inputs,
            config,
            stream_mode="messages",
            subgraphs=True,
        ):
            ## summary 랑 result 제외
            if chunk_msg != ():
                # print(f"chunk_msg : {chunk_msg} \n")
                # print(f"metadata : {metadata} \n")
                _metadata = metadata[0]
                if hasattr(_metadata, "tool_calls") and _metadata.tool_calls:
                    # Initialize tool call result
                    tool_arg = {
                        "tool_name": "",
                        "tool_result": "",
                        "tool_call_id": _metadata.tool_calls[0]["id"],
                    }
                    # Save tool name
                    tool_arg["tool_name"] = _metadata.tool_calls[0]["name"]
                    if tool_arg["tool_name"]:
                        tool_args.append(tool_arg)

                # if (
                #     hasattr(_metadata, "tool_call_chunks")
                #     and _metadata.tool_call_chunks
                # ):
                #     if len(_metadata.tool_call_chunks) > 0:  # Add None check
                #         # Accumulate tool call arguments
                #         _metadata.tool_call_chunks[0]["args"]

                if isinstance(_metadata, ToolMessage):
                    # Save tool execution results
                    current_tool_message = get_current_tool_message(
                        tool_args, _metadata.tool_call_id
                    )
                    if current_tool_message:
                        current_tool_message["tool_result"] = _metadata.content
                        with st.status(f'✅ {current_tool_message["tool_name"]}'):
                            if current_tool_message["tool_name"] == "run_python_repl":
                                print(f"metadata : {metadata} \n")
                                filename = (
                                    metadata[0].content.replace("Success: ", "").strip()
                                )
                                st.image(filename)
                                st.markdown(f"![Alt text]({filename})")
                            st.markdown(current_tool_message["tool_result"])
                if metadata[1]["langgraph_node"] == "agent":
                    if _metadata.content:
                        if agent_message is None:
                            agent_message = st.empty()
                        # Accumulate agent message
                        agent_answer += _metadata.content
                        agent_message.markdown(agent_answer)

        _tool_args = [
            tool_arg for tool_arg in tool_args if tool_arg.get("tool_result") != ""
        ]
        return container, _tool_args, agent_answer


@dataclass
class ChatMessageWithType:
    chat_message: ChatMessage
    msg_type: str
    tool_name: str


def add_message(role, message, msg_type="text", tool_name=""):
    if msg_type == "text":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(role=role, content=message),
                msg_type="text",
                tool_name=tool_name,
            )
        )
    elif msg_type == "tool_result":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(
                    role="assistant", content=format_search_result(message)
                ),
                msg_type="tool_result",
                tool_name=tool_name,
            )
        )


# 이전 대화를 출력
def print_messages():
    for message in st.session_state["messages"]:
        if isinstance(message, ChatMessageWithType):
            if message.msg_type == "text":
                st.chat_message(message.chat_message.role).write(
                    message.chat_message.content
                )
            elif message.msg_type == "tool_result":
                with st.expander(f"✅ {message.tool_name}"):
                    st.markdown(message.chat_message.content)


print_messages()
print_history()


if user_input := st.chat_input():
    add_message("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        inputs = {"question": user_input}

        container_messages, tool_args, agent_answer = stream_handler(
            container,
            st.session_state["chain"],
            inputs,
            st.session_state["config"],
        )
        # print(f"tool_args : {tool_args} \n\n")
        # print(f"agent_answer : {agent_answer}\n\n")
        # 대화기록을 저장한다.
        # add_message("user", user_input)
        for tool_arg in tool_args:
            add_message(
                "assistant",
                tool_arg["tool_result"],
                "tool_result",
                tool_arg["tool_name"],
            )
        add_message("assistant", agent_answer)
        # print(st.session_state["messages"])
