from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings




class rag_pdf:
    """
    RAG 기반 정보를 검색 결과를 반환 하는 클래스 입니다.
    """
    def __init__(self):
        """
        GoogleNews 클래스를 초기화합니다.
        base_url 속성을 설정합니다.
        """
        self.loader = PyPDFLoader("../data/SPRI_AI_Brief_2023년12월호_F.pdf")
        self.text_spitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.split_docs = self.loader.load_and_split(self.text_spitter)

    def _get_embeddings(self,embedding='openAI'):
        if embedding=='openAI':
            return OpenAIEmbeddings()
        else:
            model_name = "intfloat/multilingual-e5-large-instruct"
            hf_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda"},  # cuda, cpu
                encode_kwargs={"normalize_embeddings": True},
            )
            return hf_embeddings

    def _query(self,keyword,vector,k=3):
        self.retriever = vector.as_retriever(k=k)
        return self.retriever.invoke(keyword)

    def get_retriever(self,k=3):
        vector = FAISS.from_documents(documents=self.split_docs, embedding=self._get_embeddings())
        return vector.as_retriever(k=k)
    

    def search_by_keyword(self, keyword):
        """
        최신 뉴스를 검색합니다.

        Args:
            k (int): 검색할 뉴스의 최대 개수 (기본값: 3)

        Returns:
            List[Dict[str, str]]: URL과 내용을 포함한 딕셔너리 리스트
        """

        vector = self.get_retriever()
        return self._query(keyword,vector)

 



