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
def main():
    print("  ")
if __name__ == '__main__':
    main()