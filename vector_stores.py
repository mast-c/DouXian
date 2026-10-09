from langchain_chroma import Chroma
import config_data as config
import os
from dotenv import load_dotenv

load_dotenv()

class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
        collection_name=config.collection_name,
        embedding_function=self.embedding,
        persist_directory=config.persist_directory,
    )

    def get_retriever(self):
        """返回向量检索器, 方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})

if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(
            model=os.getenv("DASHSCOPE_EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )).get_retriever()
    res = retriever.invoke("我的体重140斤,帮我推荐一下尺码")
    print(res)