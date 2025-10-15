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

if "tmp_messages" not in st.session_state:
    st.session_state["tmp_messages"] = []


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
    messages = get_double_array(st.session_state["messages"])
    for msg in messages:
        role = list(msg.keys())[0]
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
                                filename = ai_content.replace("Success: ", "").strip()
                                st.image(filename)
                            st.markdown(ai_content)


with st.sidebar:
    clear_btn = st.button("대화내용 초기화")


if clear_btn:
    st.session_state["messages"].clear()
    st.session_state["tmp_messages"].clear()
    st.session_state["chain"] = get_graph()
    st.session_state["config"] = get_config()


if "chain" not in st.session_state:
    st.session_state["chain"] = get_graph()
    st.session_state["config"] = get_config()


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
        with st.spinner("응답 생성 중..."):
            for chunk_msg, metadata in agent_executor.stream(
                inputs,
                config,
                stream_mode="messages",
                subgraphs=True,
            ):
                # print(f" chunk_msg : \n{chunk_msg}\n")
                # print(f" metadata : \n{metadata}\n")
                try:
                    ## summary와 supervisor 화면 상에서 제거
                    if chunk_msg != () and "supervisor" not in chunk_msg[0]:

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

                        if isinstance(_metadata, ToolMessage):
                            # Save tool execution results
                            current_tool_message = get_current_tool_message(
                                tool_args, _metadata.tool_call_id
                            )
                            if current_tool_message:
                                current_tool_message["tool_result"] = _metadata.content
                                with st.status(
                                    f'✅ {current_tool_message["tool_name"]}'
                                ):
                                    if (
                                        current_tool_message["tool_name"]
                                        == "run_python_repl"
                                    ):
                                        filename = (
                                            metadata[0]
                                            .content.replace("Success: ", "")
                                            .strip()
                                        )
                                        st.image(filename)
                                    st.markdown(current_tool_message["tool_result"])
                        if metadata[1]["langgraph_node"] == "agent":
                            if _metadata.content:
                                if agent_message is None:
                                    agent_message = st.empty()
                                # Accumulate agent message
                                agent_answer += _metadata.content
                                agent_message.markdown(agent_answer)
                except Exception as e:
                    st.error(f"에러 발생: {str(e)}")
        # supervisor 노드가 직접 답변하는 경우 출력하는 로직
        try:
            past_data = agent_executor.get_state(config).values
            if len(past_data["messages"]) <= 3:
                agent_answer = past_data["answer"].content
                if agent_message is None:
                    agent_message = st.empty()
                agent_message.markdown(agent_answer)
        except Exception as e:
            st.warning(f"llm 가져오기 중 에러: {str(e)}")

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
    st.session_state["tmp_messages"].append(ChatMessage(role=role, content=message))
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
# def print_messages():
#     for message in st.session_state["messages"]:
#         if isinstance(message, ChatMessageWithType):
#             if message.msg_type == "text":
#                 st.chat_message(message.chat_message.role).write(
#                     message.chat_message.content
#                 )
#             elif message.msg_type == "tool_result":
#                 with st.expander(f"✅ {message.tool_name}"):
#                     st.markdown(message.chat_message.content)
# print_messages()

print_history()

config = st.session_state["config"]

if user_input := st.chat_input():
    add_message("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        inputs = {"question": user_input, "messages": st.session_state["tmp_messages"]}
        container_messages, tool_args, agent_answer = stream_handler(
            container,
            st.session_state["chain"],
            inputs,
            config,
        )
        st.session_state["tmp_messages"].clear()
        for tool_arg in tool_args:
            add_message(
                "assistant",
                tool_arg["tool_result"],
                "tool_result",
                tool_arg["tool_name"],
            )
        add_message("assistant", agent_answer)
        # print(st.session_state["tmp_messages"])
        # print(st.session_state["chain"].get_state(config))
