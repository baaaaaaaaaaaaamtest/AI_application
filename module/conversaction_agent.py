import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("utils"), "..")))
from module.state import *
from module.prompt import *
from module.custom_model import *
from module.base_model import *
from module.tools import *
from module.db_query import *
from langgraph.prebuilt import create_react_agent


def get_prompt_conversaction():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """ 
                You are an interactive agent that helps users with their daily conversations.  
                If you are simply talking, asking for opinions, or asking for explanations, you should answer directly, rather than calling the tool.  

                Follow these principles:
                1. Understand your intentions first, and if you don't need to use the tool, talk directly in **natural and human language.**.
                2. If your question is technical or doesn't require real-time data, calculations, visualizations, DB usage, or file access, then **only generate interactive responses.**.
                3. The purpose of the conversation is to communicate in a friendly and clear manner. Answer without being formal, but not rude or informal.
                4. When users ask their emotions, opinions, small talk, daily experiences, philosophical questions, etc., they provide **Answer based on empathy and thinking**.
                5. Keep the flow of the conversation natural, and don't end with a short counter-action, **add a word to continue the conversation.**.

                Example:
                - A user said, "The weather is nice today." → "Yes, it's really cool these days. It's the perfect day to take a walk."
                - A user said, "I'm so tired these days." → "I guess you've been having a hard time. What's the matter these days?"
                - A user asked, "Can AI understand human emotions?" → "It's not perfect yet, but there are many attempts to recognize emotional patterns. What do you think?"

                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{messages}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


def get_conversation_agent():
    llm = get_gpt()
    prompt = get_prompt_conversaction()
    return create_react_agent(
        llm,
        [],
        prompt=prompt,
        name="conversation_agent",
    )
