
# Query Rewrite 프롬프트 정의
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

"""
    PromptTemplate: 
        일반 텍스트 기반 LLM(대화형 모델이 아닌 일반 생성 모델)에서 단일 문자열 형태의 프롬프트를 만들 때 사용합니다. 
        입력값을 중괄호({})로 표기해 정보만 해당 위치에 삽입하는 방식입니다. 
        예시: 번역이나 요약 등 하나의 문장 혹은 패턴이 반복되는 사용에 적합합니다.

    ChatPromptTemplate: 
        대화형 LLM(Chat 기반 모델)에서 여러 역할(role, system/user/ai 등)로 구성된 메시지 리스트를 기반으로 대화 프롬프트를 만들 때 사용합니다. 
        메시지의 흐름(시스템 입력, 사용자 입력, AI 응답 등)을 담을 수 있고, 여러 턴(turn)에 걸친 복잡한 대화를 구성하는 데 최적화되어 있습니다.

        ** 추가 MessagesPlaceholder
        
        MessagesPlaceholder는 프롬프트 템플릿 내 특정 위치에 동적으로 메시지 목록(대화 히스토리 등)을 삽입하는 역할을 합니다.

        예를 들어, 과거 대화 내용(사용자와 AI의 메시지 리스트)을 변수로 받아서 그 위치에 자동으로 해당 메시지들이 삽입됩니다.

        템플릿을 호출할 때 여러 메시지 객체를 리스트 형태로 넣을 수 있으며, 이를 쉽게 관리하고 대화 컨텍스트를 자연스럽게 이어나가게 만듭니다.

        주로 대화형 AI에서 이전 대화 기록을 포함시켜 응답 맥락을 유지할 때 유용합니다.
"""


def get_prompt_require_infomation()->ChatPromptTemplate:
    """
        사용자 요구사항 수집을 위한 시스템 메시지 템플릿
    """
    template = """
    Your job is to gather complete and clear information from a user about the prompt template they want to create.

    You must explicitly ask for each of the following, one by one, if not provided:
    - The objective of the prompt
    - The list of variables to include in the prompt template
    - Any constraints about what the output must NOT do
    - Any requirements the output MUST satisfy

    If you cannot clearly identify any of these, politely ask the user to clarify or provide more details. 
    Do NOT guess or assume missing information.

    Only when all info is collected clearly, call the relevant tool and next step.


    [IMPORTANT] 
    Prompt generation must be done exclusively in the "prompt_generate node".
    Your conversation must be in Korean.
    The prompt you generate must be in English.
    """
    return ChatPromptTemplate(
    [
        ("system",template),
        ("placeholder", "{placeholder}"),
    ]
)

def get_prompt_assistant()->ChatPromptTemplate:
    """
        이전 대화를 모두 불러와 사용하는 유형의 챗봇
    """
    system_template = """
        You are a service support chatbot for the "Robert Chicken" system, 
        which is an automated chicken preparation and service system powered by a collaborative robot.  \n
        You can provide detailed information about system introduction, operation methods, troubleshooting, and cleaning procedures.  \n
        All responses must be professional yet customer-friendly,
        and you must always answer in Korean.  \n
    """
    return ChatPromptTemplate(
        [
            ("system",system_template),
            ("placeholder", "{messages}"),
        ]
    )

def get_prompt_persona()->ChatPromptTemplate:
    """
        가상의 환경에 놓인 사용자 
    """
    template ="""
        You are a customer of an Robert Chicken. \
        You are interacting with a user who is a customer support person. \

        Your name is james

        # Instructions:
        I recently purchased a collaborative robot-based chicken automation system from Robert Chicken. 
        I plan to use this system to start my own chicken restaurant. With the chicken automation system, 
        I aim to reduce labor costs and consistently produce the same quality of chicken to ensure customer satisfaction. 
        However, I am frustrated because operating the chicken cooking system is too difficult, 
        and malfunctions occur frequently.

        [IMPORTANT] 
        - When you are finished with the conversation, respond with a single word 'FINISHED'
        - You must speak in Korean.
    """
    return ChatPromptTemplate(
        [
            ("system",template),
            ("placeholder","{messages}")
        ]
    )

def get_prompt_answer()->PromptTemplate:
    """
        input_variables=['question','answer']\n
        Yes → 답변이 질문을 해결하거나 적절히 대답했다는 의미\n
        No → 답변이 질문을 해결하지 못했거나 엉뚱한 답변이라는 의미\n
    """
    return PromptTemplate(
        input_variables=['question','answer'],
        template="""
            # System :
            You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.

            # Human :
            User question: 
            \n\n 
            {question} 
            \n\n 
            LLM generation: 
            {answer}
        """
    )

def get_prompt_hallucination()->PromptTemplate:
    """
        input_variables=['answer','document']
        Yes → LLM의 답변이 검색된 사실에 의해 뒷받침된다.
        No → LLM의 답변이 검색된 사실에 의해 뒷받침되지 않는다. -> 재 생성
    """
    return PromptTemplate(
        input_variables=['answer','document'],
        template="""
            # System : 
            You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.

            # Human : 
            Set of facts: 
            \n\n 
            {document} 
            \n\n 
            LLM generation: 
            {answer}
        """
    )

def get_prompt_grade()->PromptTemplate:
    return PromptTemplate(
        input_variables=['question','document'],
        template="""
            # System :
            You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

            # Human : 
            Retriever Docuemnt:
            \n\n
            {document}
            \n\n
            User Question :
            \n\n
            {question}
            """
    )

def get_prompt_routing()->PromptTemplate:
    return PromptTemplate(
        input_variables=['question'],
        template="""
            # System : 

            You are an AI system that routes user questions into one of three approaches based on the question's nature:
            vectorstore: Contains information such as the DEC 2023 AI Brief Report (SPRI).
            web_search:When the user requests recent information or older data, suggest using the web_search tool to retrieve up-to-date external data. 
            This helps ensure the response is accurate and current by accessing information beyond internal sources.
            generate: Used for everyday dialogues, casual conversations, or counseling scenarios.
            Based on the user's question, please suggest the most appropriate routing approach among these three.

            # Here is the user's QUESTION that you should answer:
            {question}

            # Your final ANSWER to the user's QUESTION:
        """
    )

def get_prompt_rag()->PromptTemplate:
    """
        Argument:
        input_variables=['context', 'question']
    """
    return PromptTemplate(
    input_variables=['context', 'question'],
    template="""

        You are an AI assistant specializing in Question-Answering (QA) tasks within a Retrieval-Augmented Generation (RAG) system. 
        Your primary mission is to answer questions based on provided context or chat history.
        Ensure your response is concise and directly addresses the question without any additional narration.

        ###

        Your final answer should be written concisely (but include important numerical values, technical terms, jargon, and names), followed by the source of the information.

        # Steps

        1. Carefully read and understand the context provided.
        2. Identify the key information related to the question within the context.
        3. Formulate a concise answer based on the relevant information.
        4. Ensure your final answer directly addresses the question.
        5. List the source of the answer in bullet points, which must be a file name (with a page number) or URL from the context. Omit if the source cannot be found.

        # Output Format:
        [Your final answer here, with numerical values, technical terms, jargon, and names in their original language]

        ###

        Remember:
        - It's crucial to base your answer solely on the **PROVIDED CONTEXT**. 
        - DO NOT use any external knowledge or information not present in the given materials.
        - If you can't find the source of the answer, you should answer that you don't know.

        ###

        # Here is the user's QUESTION that you should answer:
        {question}

        # Here is the CONTEXT that you should use to answer the question:
        {context}

        # Your final ANSWER to the user's QUESTION:
        """
    )

def get_prompt_re_write()->PromptTemplate:
    return  PromptTemplate(
    input_variables=["question"],
    template="""
        Reformulate the given question to enhance its effectiveness for vectorstore retrieval.
        - Analyze the initial question to identify areas for improvement such as specificity, clarity, and relevance.
        - Consider the context and potential keywords that would optimize retrieval.
        - Maintain the intent of the original question while enhancing its structure and vocabulary.

        # Steps

        1. **Understand the Original Question**: Identify the core intent and any keywords.
        2. **Enhance Clarity**: Simplify language and ensure the question is direct and to the point.
        3. **Optimize for Retrieval**: Add or rearrange keywords for better alignment with vectorstore indexing.
        4. **Review**: Ensure the improved question accurately reflects the original intent and is free of ambiguity.

        # Output Format

        - Provide a single, improved question.
        - Do not include any introductory or explanatory text; only the reformulated question.

        # Examples

        **Input**: 
        "Hi, My name is kay"

        **Output**: 
        "Hi, nice to meet you, my name is kay. Could you help me?"

        **Input**: 
        "What are the benefits of using renewable energy sources over fossil fuels?"

        **Output**: 
        "How do renewable energy sources compare to fossil fuels in terms of benefits?"

        **Input**: 
        "How does climate change impact polar bear populations?"

        **Output**: 
        "What effects does climate change have on polar bear populations?"

        # Notes

        - Ensure the improved question is concise and contextually relevant.
        - Avoid altering the fundamental intent or meaning of the original question.


        [REMEMBER] Re-written question should be in the same language as the original question.

        # Here is the original question that needs to be rewritten:
        {question}
        """,
    
)





# 프롬프트를 생성하는 메타 프롬프트 정의(OpenAI 메타 프롬프트 엔지니어링 가이드 참고)
META_PROMPT = """Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

[User given variables should be wrapped in {{brackets}}]

<Question>
{{question}}
</Question>

<Answer>
{{answer}}
</Answer>

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]

# Based on the following requirements, write a good prompt template:

{reqs}
"""


# 프롬프트 생성을 위한 메시지 가져오기 함수
# 도구 호출 이후의 메시지만 가져옴
def get_prompt_messages(messages: list):
    # 도구 호출 정보를 저장할 변수 초기화
    tool_call = None
    # 도구 호출 이후의 메시지를 저장할 리스트 초기화
    other_msgs = []
    # 메시지 목록을 순회하며 도구 호출 및 기타 메시지 처리
    for m in messages:
        # AI 메시지 중 도구 호출이 있는 경우 도구 호출 정보 저장
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        # ToolMessage는 건너뜀
        # elif isinstance(m, ToolMessage):
        #     continue
        # tool_call 객체가 실제로 존재하는지를 확인
        elif tool_call is not None:
            other_msgs.append(m)
    print("other_msgs :           ",other_msgs)
    # 시스템 메시지와 도구 호출 이후의 메시지를 결합하여 반환
    return [SystemMessage(content=META_PROMPT.format(reqs=tool_call))] + other_msgs
