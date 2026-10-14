import os
import re
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
import re
import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  #RRF
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import config_data as config
import dashscope
from http import HTTPStatus


load_dotenv()
md5_path = config.md5_path


def check_md5(md5_str: str):
    """检查文档内容是否已处理。"""
    if not os.path.exists(md5_path):
        return False

    with open(md5_path, "r", encoding="utf-8") as f:
        return any(line.strip() == md5_str for line in f)


def save_md5(md5_str: str):
    with open(md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str: str, encoding="utf-8"):
    return hashlib.md5(input_str.encode(encoding)).hexdigest()



def tokenize_for_bm25(text: str) -> list[str]:
    """中文分词，额外保留 ERR-1009 之类的完整编号。"""

    words = [
        word.lower()
        for word in jieba.lcut_for_search(text)
        if any(char.isalnum() for char in word)
    ]

    codes = re.findall(
        r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+",
        text
    )

    words.extend(code.lower() for code in codes)

    return words
class KnowledgeBaseService:

    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(
                model=os.getenv("DASHSCOPE_EMBEDDING_MODEL"),
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            ),
            persist_directory=config.persist_directory,
        )

        # 原有 Recursive 切分器：不改参数
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
            keep_separator="end",
        )

        # 新增：Markdown 结构切分器
        
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"), 
            ],
            strip_headers=True,
        )

    # 新增：第一步结构切分
    def split_by_structure(
        self, data: str, filename: str
    ) -> list[Document]:

        text = data.replace("\r\n", "\n").replace("\r", "\n").strip()

        if not text:
            return []

        # 根据内容识别 Markdown 标题
        has_md_headers = re.search(
            r"(?m)^#{1,3}[ \t]+\S", text
        )

        # Markdown 文件，或者转换成 Markdown 的 PDF/Word
        if filename.lower().endswith(
            (".md", ".markdown")
        ) or has_md_headers:
            return self.md_splitter.split_text(text)

        # 普通文本按空行分段
        paragraphs = re.split(r"\n[ \t]*\n+", text)

        return [
            Document(page_content=p.strip())
            for p in paragraphs
            if p.strip()
        ]

    
    def split_into_chunks(self, data: str, filename: str) -> list[Document]:
        """结构切分 + Recursive 二次切分，并为每个 Chunk 补充章节标题。"""

        sections = self.split_by_structure(data, filename)
        knowledge_docs = []

        for section in sections:
            # 获取当前章节路径
            title_path = " / ".join(
                section.metadata[key]
                for key in ("h1", "h2", "h3","h4","h5")
                if section.metadata.get(key)
            )

            prefix = f"章节：{title_path}\n" if title_path else ""

            # 为章节前缀预留空间，确保最终 Chunk 不超过 chunk_size
            body_size = config.chunk_size - len(prefix)

            if body_size <= config.chunk_overlap:
                raise ValueError("章节标题过长，请增大 chunk_size 或缩短标题")

            # 超出阈值时进行递归切分
            if (
                len(section.page_content) > config.max_split_char_number
                or len(section.page_content) > body_size
            ):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=body_size,
                    chunk_overlap=config.chunk_overlap,
                    separators=config.separators,
                    length_function=len,
                    keep_separator="end",
                )
                chunks = splitter.split_documents([section])
            else:
                chunks = [section]

            # 给每个 Chunk 添加章节上下文
            for chunk in chunks:
                chunk.page_content = prefix + chunk.page_content
                knowledge_docs.append(chunk)

        return knowledge_docs
    
    def add_metadata(
        self,
        knowledge_docs: list[Document],
        filename: str
    ) -> list[Document]:
        """为最终 Chunk 添加来源、章节路径和位置信息。"""

        # 同一个 filename 对应稳定的文档 ID
        doc_id = hashlib.sha256(filename.encode("utf-8")).hexdigest()

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chunk_total = len(knowledge_docs)

        for index, doc in enumerate(knowledge_docs):

            # 组合章节标题
            titles = [
                doc.metadata.get(key, "")
                for key in ("h1", "h2", "h3","h4","h5")
            ]

            section_path = " / ".join(
                title for title in titles if title
            )

            # 保留原有 h1、h2、h3 信息
            doc.metadata = {
                **doc.metadata,
                "source": filename,
                "doc_id": doc_id,
                "chunk_index": index,
                "chunk_total": chunk_total,
                "section_path": section_path,
                "create_time": created_at,
                "operator": "jlp",
            }

        return knowledge_docs
    
    def upload_by_str(self, data: str, filename: str) -> str:
        """按来源更新文档：相同 Chunk 不重复入库，变化后替换旧 Chunk。"""
        # 统一文档名称，避免相对路径不同导致重复入库
        filename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
        if not isinstance(data, str) or not data.strip():
            return "[跳过]文档内容为空"
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename 不能为空")
        filename = filename.strip()

        # 1. 复用之前已经实现的切分与 Metadata 方法
        knowledge_docs = self.add_metadata(
            self.split_into_chunks(data, filename), filename
        )
        if not knowledge_docs:
            return "[跳过]未产生有效 Chunk"

        # 2. 为每个 Chunk 生成确定性的唯一 ID
        ids = []
        for doc in knowledge_docs:
            raw = (
                f"{doc.metadata['doc_id']}:"
                f"{doc.metadata['chunk_index']}:"
                f"{doc.page_content}"
            )
            chunk_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            doc.metadata["chunk_id"] = chunk_id
            ids.append(chunk_id)

        # 3. 查找该来源以前入库的所有 Chunk
        # 也兼容之前由 Chroma 自动生成 UUID 的旧数据
        old_ids = set(
            self.chroma.get(
                where={"source": filename},
                include=["metadatas"],
            )["ids"]
        )
        new_ids = set(ids)

        # 4. 内容和切分结果均未变化，则无需重新向量化
        if old_ids == new_ids:
            return f"[跳过] {filename} 内容未变化（{len(ids)} 个 Chunk）"

        # 5. 先写入新版本，再清理不再使用的旧 Chunk
        self.chroma.add_texts(
            texts=[doc.page_content for doc in knowledge_docs],
            metadatas=[doc.metadata for doc in knowledge_docs],
            ids=ids,
        )

        stale_ids = sorted(old_ids - new_ids)
        if stale_ids:
            self.chroma.delete(ids=stale_ids)

        return (
            f"[成功] {filename}：写入 {len(ids)} 个 Chunk，"
            f"清理 {len(stale_ids)} 个旧 Chunk"
        )
    
    def build_bm25_index(self, batch_size: int = 200) -> int:
        """从 Chroma 中读取已有 Chunk，构建内存 BM25 索引。"""

        if batch_size < 1:
            raise ValueError("batch_size 必须大于 0")

        documents = []
        offset = 0

        while True:
            batch = self.chroma.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )

            ids = batch["ids"]
            if not ids:
                break

            texts = batch.get("documents") or [None] * len(ids)
            metadatas = batch.get("metadatas") or [None] * len(ids)

            for chunk_id, text, metadata in zip(ids, texts, metadatas):
                if not text or not text.strip():
                    continue

                metadata = dict(metadata or {})
                metadata["chunk_id"] = chunk_id

                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

            offset += len(ids)

            if len(ids) < batch_size:
                break

        self.bm25_retriever = (
            BM25Retriever.from_documents(
                documents,
                preprocess_func=tokenize_for_bm25,
                k=5,
            )
            if documents else None
        )

        return len(documents)


    def search_bm25(self, query: str) -> list[Document]:
        """通过 BM25 检索关键词相关的 Chunk。"""

        if not isinstance(query, str) or not query.strip():
            return []

        if not hasattr(self, "bm25_retriever"):
            raise RuntimeError("请先执行 build_bm25_index()")

        if self.bm25_retriever is None:
            return []

        return self.bm25_retriever.invoke(query)
        
    def build_hybrid_retriever(self):
        """组合 Chroma 向量检索和 BM25 检索。"""
        if not hasattr(self, "bm25_retriever"):
            raise RuntimeError("请先调用 build_bm25_index()")

        if self.bm25_retriever is None:
            raise ValueError("BM25 索引为空")

        vector_retriever = self.chroma.as_retriever(
            search_kwargs={"k": 20}
        )

        self.bm25_retriever.k = 20

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                vector_retriever,
                self.bm25_retriever,
            ],
            weights=[0.5, 0.5],
            id_key="chunk_id",
        )
        print(f"混合检索器已构建: 向量 k=20, BM25 k=20, 权重={self.hybrid_retriever.weights}")

    def search_hybrid(self, query: str, k: int = 5) -> list[Document]:
        if not isinstance(query, str) or not query.strip():
            return []
        if k <= 0:
            return []
        if not hasattr(self, "hybrid_retriever") or self.hybrid_retriever is None:
            raise RuntimeError("请先调用 build_hybrid_retriever()")

        return self.hybrid_retriever.invoke(query)[:k]
    
    def search_with_rerank(
        self, query: str, recall_k: int = 10, top_n: int = 3
    ) -> list[Document]:
        """RRF 融合 → DashScope 重排；保留原始 Document 的元数据。"""
        if not isinstance(query, str) or not query.strip():
            return []
        if recall_k < 1 or top_n < 1:
            raise ValueError("recall_k 和 top_n 都必须为正整数")

        candidates = self.search_hybrid(query, k=recall_k)
        if not candidates:
            return []

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY")
        base_url = os.getenv("DASHSCOPE_HTTP_BASE_URL")
        if base_url:
            dashscope.base_http_api_url = base_url

        response = dashscope.TextReRank.call(
            model="qwen3-rerank",
            api_key=api_key,
            query=query,
            documents=[doc.page_content for doc in candidates],
            top_n=min(top_n, len(candidates)),
        )
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"Rerank重拍失败: {response.code} {response.message}"
            )

        items = (response.output or {}).get("results", [])
        if not items:
            raise RuntimeError("重排未返回任何结果")

        ranked = []
        seen = set()
        for item in items:
            idx = int(item["index"])
            if idx < 0 or idx >= len(candidates) or idx in seen:
                raise ValueError(f"无效或重复的重排索引: {idx}")
            seen.add(idx)
            doc = candidates[idx]
            ranked.append(Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(item["relevance_score"])}
            ))

        return ranked[:top_n]
def main():
    print("")
if __name__ == "__main__":
   main()