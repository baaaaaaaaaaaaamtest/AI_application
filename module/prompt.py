# Query Rewrite 프롬프트 정의
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
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


def get_prompt_routing_node() -> ChatPromptTemplate:
    """
    web search, sql, pdf 등 어떤 agent 를 선택할지 결정하게 지원하는 프롬프트

    If the remaining requirements no longer exist,
    prefix your response with FINAL ANSWER so the team knows to stop.
    """
    template = """
        You are an AI assistant
        If you need music information, search Genie Music and get information
        
        You have some agents :
        1. The first agent is the agent for web search. Most requests can be resolved through web search

        2. The second agent is the Database agent. To provide music services, we can use a database that can manage, inquire, and manage the sales performance of artists and music.
        
        3. The third agent can be visualized through data analysis requests and comparisons. By generating the Python code, it is possible to generate bar charts, donut charts, and line graphs based on this.
         
        4. other 'conversation_agent'.Agent that conducts a normal conversation. 
        It is very important to identify the user's requirements and use an agent that fits the situation
        
        # Next Task : 
        {next_task}

        # Important :
        We are going to solve the given `next_task` using agent.
        Make sure to solve the problem without asking the user for additional information.
        Write your response in Korean.
    """
    return ChatPromptTemplate.from_messages(
        [("system", template), ("placeholder", "{placeholder}")]
    )


def get_prompt_start_node() -> ChatPromptTemplate:
    template = """
        You have some agents :

        1. The first agent is the agent for web search. Most requests can be resolved through web search

        2. The second agent is pdf loader, which is used to search for pre-vectorized data or internal data

        3. The last agent is the Database agent. To provide music services, we can use a database that can manage, inquire, and manage the sales performance of artists and music.

        It is very important to identify the user's requirements and use an agent that fits the situation
        
        # Important :
        If the remaining requirements no longer exist,
        prefix your response with FINAL ANSWER so the team knows to stop.
        Write your response in Korean.
    """
    return ChatPromptTemplate.from_messages(
        [("system", template), ("placeholder", "{placeholder}")]
    )


def get_prompt_query_check() -> ChatPromptTemplate:
    template = """You are a SQL expert with a strong attention to detail.
    Double check the SQLite query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

    You will call the appropriate tool to execute the query after running this check."""

    return ChatPromptTemplate.from_messages(
        [("system", template), ("placeholder", "{placeholder}")]
    )


def get_prompt_multi_chart() -> ChatPromptTemplate:

    template = """
            # System : 
            You are a helpful AI assistant, collaborating with other assistants.  
            You are working with a web search generator colleague.
  
            
            # Step :
            1. Receive information from another tool and based on that information, generate charts. 
            2. When If your generate chart, you need more additional data is requests are made to a web tool colleague. 
            3. Then, using this information, the Python REPL tool is used to generate the charts.

            If you are unable to fully answer, that's OK, another assistant with different tools
            will help where you left off. Execute what you can to make progress.
        
            # Human :
            {messages}
            
            # Important :
            If you or any of the other assistants have the final answer or deliverable,
            prefix your response with FINAL ANSWER so the team knows to stop.
            Write your response in Korean.
          
        """

    return ChatPromptTemplate.from_template(template)


def get_prompt_web() -> ChatPromptTemplate:
    template = """
            # System : 
            You are an AI agent specialized in web search. 
            Your mission is to understand users' queries, generate the most efficient search queries, and provide accurate, reliable, and up-to-date information.

            Follow these guidelines:

            1. **Question Interpretation**:

            Clearly analyze the user's intent and extract core keywords.
            Restructure ambiguous questions into search-friendly terms.

            2. **Search Query Generation**:
            - Create search terms that are as concise and relevant as possible.
            - Consider synonyms and related keywords to experiment with different queries.
            
            3. **Search Results Summary**:
            - Concisely summarize the key facts, figures, and cited sources from the search results.
            - Eliminate unnecessary advertisements, commercial noise, and irrelevant sentences.
            - Ensure your answers are up-to-date and reliable.

            4. **Output Format**:
            - Concise summary
            - Bulleted summary of key points

            Always provide objective, source-based information.
            Don't guess; only provide verifiable information.
            
        
            # Human :
            {current_steps}
            
            # Important :
            Write your response in Korean.
          
        """

    return ChatPromptTemplate.from_template(template)


def get_prompt_multi_web() -> ChatPromptTemplate:

    template = """
            # System : 
            You are a helpful AI assistant, collaborating with other assistants.  
            You are working with a pdf loader agent colleague.
        
            # Human :
            {messages}
            
            # Important :
            If you or any of the other assistants have the final answer or deliverable,
            prefix your response with FINAL ANSWER so the team knows to stop.
            Write your response in Korean.
          
        """

    return ChatPromptTemplate.from_template(template)


def get_prompt_multi_loader() -> ChatPromptTemplate:

    template = """
            # System : 
            You are a helpful AI assistant, collaborating with other assistants.  
            You are working with a web search agent colleague.

            # Human :
            {messages}
            
            # Important :
            If you or any of the other assistants have the final answer or deliverable,
            prefix your response with FINAL ANSWER so the team knows to stop.
            Write your response in Korean.
          
        """

    return ChatPromptTemplate.from_template(template)


def get_prompt_generate_markdown() -> ChatPromptTemplate:

    return ChatPromptTemplate.from_template(
        """You are given the objective and the previously done steps. Your task is to generate a final report in markdown format.
    Final report should be written in professional tone.

    Your objective was this:

    {input}

    Your previously done steps(question and answer pairs):

    {past_steps}

    Generate a final report in markdown format. Write your response in Korean."""
    )


def get_prompt_replanner() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """
        # System :
        If you determine that you already objective achieve the entire remaining plan, you can remove all remaining plans.
        You can remove any remaining plans if they overlap or are unnecessary.


        # Your objective was this:
        {input}

        # Your remaining plan was this:
        {plan}

        # You have currently done the follow steps:
        {past_steps}

        Answer in Korean."""
    )


# def get_prompt_replanner()->ChatPromptTemplate:
#     # 계획을 재수립하기 위한 프롬프트 정의
#     """
#         input_valuable=['input','plan','past_steps']
#     """
#     return ChatPromptTemplate.from_template(
#         """\
#     This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
#     The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

#     Your objective was this:
#     {input}

#     Your original plan was this:
#     {plan}

#     You have currently done the follow steps:
#     {past_steps}

#     Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that.

#     Otherwise, We will continue to keep our initial plans unchanged. Do not return previously done steps as part of the plan.

#     Answer in Korean."""
#     )


def get_prompt_agent() -> ChatPromptTemplate:
    """
    assistant 전용 prompt
    """
    templete = """
        # System :
        You are a helpful assistant. Almost out service user live in Seoul, South korea

        1. If you want search weather, you can choice use about `weather_search_tool`

        2. If you want search artist infomation, you can choice use about `artist_news_search_tool`

        3. If you do a query in DB, make sure to use `sql_db_list_tables` first and then use `sql_db_schema` before creating query

        # Important :
        Answer in Korean.
        
    """
    return ChatPromptTemplate.from_messages(
        [("system", templete), ("human", "{messages}")]
    )


def get_prompt_music_planner() -> ChatPromptTemplate:
    """
    주어진 question을 기반으로 step 별 계획을 세우는 prompt
    value = 'messages'
    """
    templete = """
   Infomation : 
    I live in Seoul,KR
    
    System : 
    you must call `PlanListModel`
    If it's a simple conversation, I'm not going to plan it.
    If you don't need a plan, output the question in `PlanListModel.steps`.
    For the given objective, come up with a simple step by step plan. \
    Do not set steps unnecessarily or too detailed.
    The result of the final step should be the final answer. 

    Example :
    Question: Please recommend a song that fits today's weather
    Step 1. : 
    Step 2. :

    Answer : 
    stpes : ["Sample 1","Sample 2"]
    If you don't need a plan, output the question in `steps`.
    If there is no plan, output the question in `steps`.
    Answer in Korean."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", templete),
            ("placeholder", "{messages}"),
        ]
    )


def get_prompt_planner() -> ChatPromptTemplate:
    """
    주어진 question을 기반으로 step 별 계획을 세우는 prompt
    value = 'messages'
    """
    templete = """
    System : 
    For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
    Answer in Korean."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", templete),
            ("placeholder", "{messages}"),
        ]
    )


def get_prompt_require_infomation() -> ChatPromptTemplate:
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
            ("system", template),
            ("placeholder", "{placeholder}"),
        ]
    )


def get_prompt_assistant() -> ChatPromptTemplate:
    """
    이전 대화를 모두 불러와 사용하는 유형의 챗봇
    """
    system_template = """
        You are a helpful assistant. Almost out service user live in Seoul, South korea
        you must always answer in Korean.  \n
    """
    return ChatPromptTemplate(
        [
            ("system", system_template),
            ("placeholder", "{messages}"),
        ]
    )


def get_prompt_persona() -> ChatPromptTemplate:
    """
    가상의 환경에 놓인 사용자
    """
    template = """
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
    return ChatPromptTemplate([("system", template), ("placeholder", "{messages}")])


def get_prompt_answer():
    prompt = """ 
    You are a DB expert
    Print your final answer based on the query you ran previously

    # select:
    Map the information obtained from the select execution so that the user can easily understand it

    # User Request : 
    {current_steps}
    
    # Summary of conversation earlier :
    {summary}

    # placeholder:
    {messages}

    """
    return ChatPromptTemplate.from_template(prompt)


def get_prompt_hallucination() -> PromptTemplate:
    """
    input_variables=['answer','document']
    Yes → LLM의 답변이 검색된 사실에 의해 뒷받침된다.
    No → LLM의 답변이 검색된 사실에 의해 뒷받침되지 않는다. -> 재 생성
    """
    return PromptTemplate(
        input_variables=["answer", "document"],
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
        """,
    )


def get_prompt_grade() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question", "document"],
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
            """,
    )


def get_prompt_routing() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question"],
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
        """,
    )


def get_prompt_rag() -> PromptTemplate:
    """
    Argument:
    input_variables=['context', 'question']
    """
    return PromptTemplate(
        input_variables=["context", "question"],
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
        """,
    )


def get_prompt_re_write() -> PromptTemplate:
    """
    input_variables=["question"]
    """
    return PromptTemplate(
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
    print("other_msgs :           ", other_msgs)
    # 시스템 메시지와 도구 호출 이후의 메시지를 결합하여 반환
    return [SystemMessage(content=META_PROMPT.format(reqs=tool_call))] + other_msgs


def get_prompt_relevant_query():
    prompt = """ 
    # System :
    You are a database expert
    Determines the association between user requests and query execution results
    Please answer yes or no by confirming that the answer is relevant to the user's request

    # User Request:
    {current_steps}
    # AI Answer Query:
    {query}

    # Important : 
    response  is only `yes` or `no`
    """

    return ChatPromptTemplate.from_template(prompt)


def get_prompt_query_gen() -> ChatPromptTemplate:
    prompt = """You are a SQL expert with a strong attention to detail.

    You can define SQL queries, analyze queries results and interpretate query results to response an answer.

    Read the messages bellow and identify the user question, table schemas, query statement and query result, or error if they exist.

    1. If there's not any query result that make sense to answer the question, create a syntactically correct SQLite query to answer the user question. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    2. If you create a query, response ONLY the query statement. For example, "SELECT id, name FROM pets;"

    3. If a query was already executed, but there was an error. Response with the same error message you found. For example: "Error: Pets table doesn't exist"

    4. If a query was already executed successfully interpretate the response and answer the questio following this pattern: Answer: <<question answer>>. For example: "Answer: There three cats registered as adopted"

    5. Please add a semicolon (;) at the end of your SQL query.

    6. If you create a select query, please create information such as some columns of the object and basic names that a person can analyze

    7. If you run select, you must also include additional information about the unique id information as a name, etc..
    
    # User Request : 
    {current_steps}

    # placeholder:
    {placeholder}

    """
    return ChatPromptTemplate.from_template(prompt)


def get_prompt_final():
    prompt = """ 
        # System :
        You are AI Asistant
        You can answer two types

        1. If Agent to be used is `conversation_agent`, general conversations, recommendations, directional suggestions, etc. can be carried out 
        2. If Agent to be used is not a `conversation_agent`, you must ask whether you want to use that Agent or not.
        example) Do you want to proceed using db_agent?

        And if the plan does not exist anymore, please synthesize the previous content and generate the final answer
        The final answer should always be the final result of solving the problem.

        # To be processed in this step : 
        {current_steps}
        
        # Agent to be used :
        {next_agent}

        # Plan :
        {plan}

        
        Write your response in Korean.
        """
    return ChatPromptTemplate.from_template(prompt)