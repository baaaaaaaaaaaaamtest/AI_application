import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("utils"), "..")))
from langchain_core.tools import tool
from module.utils import *
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate


def get_prompt_db_query():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """ 
                You are a DB Expert AI assistant. 
                
                It is most effective to use the tools in the following order to obtain database information.

                First, call 'get_all_table_tool' to get a list of available databases.

                Then, use only the database list you obtained to select the required table.

                Pass 'get_one_table_schema_tool' to obtain detailed schema information for all selected tables. Multiple table information can be requested at once.

                Use the 'get_query_gen_tool' tool to create a query using the schema information obtained in the previous step.

                Finally, based on the generated query, you can run it in 'db_query_tool' and see the results
                
                # Important : 
                After executing a `get_all_table_tool` or `get_one_table_schema_tool` or `db_query_tool`, format the final output as a neat Markdown table with column headers and aligned cells.
                If empty, respond with “No data found.”
                """,
            ),
            ("human", "{messages}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


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
    {user_input}

    """
    return ChatPromptTemplate.from_template(prompt)


@tool
def get_all_table_tool():
    """
    You can use this tool to obtain information about all tables that exist in the database.
    """
    llm = get_gpt()
    tools = get_db_tool(llm)
    sql_db_list_tables = next(
        tool for tool in tools if tool.name == "sql_db_list_tables"
    )
    return sql_db_list_tables.invoke("")


@tool
def get_one_table_schema_tool(target_tables_name):
    """
    You must bring the schema information of the table you want to use to safely perform inquiries
    When you get a table schema, you can request multiple table information at once.
    When creating target_tables_name, be sure to refer to the 'sql_db_list_tables' list.

    Args:
    target_tables_name (dict): 사용자의 요구사항을 분석하여 필요한 테이블 리스트
    """
    llm = get_gpt()
    tools = get_db_tool(llm)
    sql_db_schema = next(tool for tool in tools if tool.name == "sql_db_schema")
    llm_with_schema = llm.bind_tools([sql_db_schema], tool_choice="sql_db_schema")
    response = llm_with_schema.invoke(target_tables_name)
    llm_gen_input = response.tool_calls[0]["args"]
    return sql_db_schema.invoke(llm_gen_input)


@tool
def get_query_gen_tool(history):
    """
    A tool that generates queries that can resolve requests using database list information and table schema information that you want to use in the previous step
    The answer must generate a query to be requested to the database.
    Args:
    history (str): 사용자 요구사항과 이전 단계에서 얻은 여러 테이블의 스키마 정보 활용 쿼리 작성
    """
    prompt = get_prompt_query_gen()
    llm = get_gpt()
    query_gen_llm = prompt | llm
    return query_gen_llm.invoke({"user_input": history})


@tool
def db_query_tool(query: str) -> str:
    """
    Run SQL queries against a database and return results
    Returns an error message if the query is incorrect
    If an error is returned, rewrite the query, check, and retry

    Args:
        query (str):  get_query_gen_tool에서 생성한 쿼리 실행
    """
    # 쿼리 실행
    db = get_db()
    result = db.run_no_throw(query)

    # 오류: 결과가 없으면 오류 메시지 반환
    if "Error" in result:
        return f"Error: {result} \n\n . Please rewrite your query and try again."
    # 정상: 쿼리 실행 결과 반환
    elif not result:
        return "Success: value is None"
    else:
        return f"Success: {result}"


# 상태 초기값 생성 예
def get_db_agent():

    llm = get_gpt()
    prompt = get_prompt_db_query()
    return create_react_agent(
        llm,
        [
            # instruction_node,
            # get_table_list_tool,
            get_all_table_tool,
            get_one_table_schema_tool,
            get_query_gen_tool,
            db_query_tool,
        ],
        prompt=get_prompt_db_query(),
        name="db_agent",
    )
