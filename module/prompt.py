
# Query Rewrite 프롬프트 정의
from langchain_core.prompts import PromptTemplate

def get_rag_pt()->PromptTemplate:
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

        **Source**(Optional)
        - (Source of the answer, must be a file name(with a page number) or URL from the context. Omit if you can't find the source of the answer.)
        - (list more if there are multiple sources)
        - ...

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

def get_re_write_pt()->PromptTemplate:
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