import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
import os, sys

sys.path.append(os.path.abspath("C:\\Users\\ansgy\\IdeaProjects\\AI_application"))

from module.langGraph import *

st.set_page_config(page_title="K ChatGPT 💬", page_icon="💬")
st.title("K ChatGPT 💬")

start_langsmith("streamlit_01_practice")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


with st.sidebar:
    clear_btn = st.button("대화내용 초기화")


if clear_btn:
    retriever = st.session_state["messages"].clear()
    st.session_state["chain"] = get_graph()
    st.session_state["config"] = get_config()

print_history()


if "chain" not in st.session_state:
    st.session_state["chain"] = get_graph()
    st.session_state["config"] = get_config()


if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        chat_container = st.empty()
        ai_answer = ""
        inputs = {"question": user_input}
        for step, metadata in st.session_state["chain"].stream(
            input=inputs,
            config=st.session_state["config"],
            stream_mode="messages",
            subgraphs=True,
        ):
            
            # if (
            #     metadata[0].content != "Transferring back to supervisor"
            #     and metadata[0].content != "Successfully transferred back to supervisor"
            #     and metadata[0].content != ""
            # ):
            if step != ():
                ai_answer += metadata[0].content
                chat_container.markdown(ai_answer)
        add_history("ai", ai_answer)

        # for step, metadata in st.session_state["chain"].stream(
        #     input=inputs, config=st.session_state["config"], stream_mode="messages"
        # ):
        #     # print(f"main : {step} \n")
        #     # print(f"main_metadata: {metadata} \n\n")
        #     if (
        #         metadata["langgraph_node"] != "summary_node"
        #         and (text := step.text())
        #         and step.name != "transfer_back_to_supervisor"
        #         and step.content != "Transferring back to supervisor"
        #     ):

        #         ai_answer += step.content
        #         ai_answer += "\n\n"
        #         chat_container.markdown(ai_answer)
        # add_history("ai", ai_answer)

        # stream_response = st.session_state["chain"].stream(
        #     config=st.session_state["config"],
        #     input={"question": user_input},
        #     stream_mode="values",
        # )  # 문서에 대한 질의
        # for chunk in stream_response:
        #     print(chunk)
        #     messages = chunk.get("messages", [])
        #     if len(messages) > 1:
        #         ai_answer += messages[-1].content
        #         ai_answer += "\n\n\n"
        #         chat_container.markdown(ai_answer)
        # add_history("ai", ai_answer)
