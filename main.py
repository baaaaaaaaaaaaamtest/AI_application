import streamlit as st
import os, sys

sys.path.append(os.path.abspath("C:\\Users\\ansgy\\IdeaProjects\\AI_application"))
from module.langGraph import *
from module.handler import *

st.set_page_config(page_title="K ChatGPT 💬", page_icon="💬")
st.title("K ChatGPT 💬")

start_langsmith("streamlit_01_practice")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "tmp_messages" not in st.session_state:
    st.session_state["tmp_messages"] = []


if "chain" not in st.session_state:
    st.session_state["chain"] = get_graph()
    st.session_state["config"] = get_config()


with st.sidebar:
    clear_btn = st.button("대화내용 초기화")

if clear_btn:
    st.session_state["messages"].clear()
    st.session_state["tmp_messages"].clear()
    st.session_state["chain"] = get_graph()
    st.session_state["config"] = get_config()

print_history(st)

if user_input := st.chat_input():
    config = st.session_state["config"]
    add_message(st, "user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        inputs = {"question": user_input, "messages": st.session_state["tmp_messages"]}
        container_messages, tool_args, agent_answer = stream_handler(
            st,
            container,
            st.session_state["chain"],
            inputs,
            config,
        )
        st.session_state["tmp_messages"].clear()
        for tool_arg in tool_args:
            add_message(
                st,
                "assistant",
                tool_arg["tool_result"],
                "tool_result",
                tool_arg["tool_name"],
            )
        add_message(st, "assistant", agent_answer)
        # print(st.session_state["tmp_messages"])
        # print(st.session_state["chain"].get_state(config))
