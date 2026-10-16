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
from langchain_classic.retrievers import EnsembleRetriever  # RRF
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import config_data as config
import dashscope
from http import HTTPStatus
from collections import defaultdict, deque
load_dotenv()


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
        # 新增：Markdown 结构切分器
        # 把标题层级和内容解析为结构化的 metadata（如 {"Header 1": "第一章", "Header 2": "1.1 概述"}）
        # 将这个 metadata 附加到该标题下所有后续 Chunk 上
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
            r"(?m)^#{1,5}[ \t]+\S", text
        )

        # Markdown 文件，或者转换成 Markdown 的 PDF/Word
        if filename.lower().endswith(
            (".md", ".markdown")
        ) or has_md_headers:
            return self.md_splitter.split_text(text)
            """ # Python 编程基础
            ## 什么是 Python
            Python 是一种高级编程语言。
            
            ## Python 的应用领域
            Python 广泛应用于 Web 开发、数据分析等.
            [
                Document(
                    page_content="Python 是一种高级编程语言。",
                    metadata={"H1": "Python 编程基础", "H2": "什么是 Python"}
                ),
                Document(
                    page_content="Python 广泛应用于 Web 开发、数据分析等。",
                    metadata={"H1": "Python 编程基础", "H2": "Python 的应用领域"}
                )
            ]"""

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
                for key in ("h1", "h2", "h3", "h4", "h5")
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
                for key in ("h1", "h2", "h3", "h4", "h5")
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
        """按 Chunk 正文增量更新，最大限度复用已有 Embedding。"""

        if not isinstance(data, str) or not data.strip():
            return "[跳过]文档内容为空"

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename 不能为空")

        # 统一 source，避免同一文件因路径不同重复入库
        filename = filename.strip().replace("\\", "/").rsplit("/", 1)[-1]

        if not filename:
            raise ValueError("filename 不能为空")

        # 1. 保留原有切分与 Metadata 生成逻辑
        knowledge_docs = self.add_metadata(
            self.split_into_chunks(data, filename),
            filename,
        )

        if not knowledge_docs:
            return "[跳过]未产生有效 Chunk"

        # 2. 查询这份文档以前入库的正文及 Metadata
        old_data = self.chroma.get(
            where={"source": filename},
            include=["documents", "metadatas"],
        )

        old_ids = old_data["ids"]
        old_texts = old_data["documents"] or []
        old_metadatas = old_data["metadatas"] or []

        if not (
            len(old_ids) == len(old_texts) == len(old_metadatas)
        ):
            raise RuntimeError("Chroma 返回的数据数量不一致")

        # 3. 建立正文 -> 旧记录的映射
        # deque 用于安全处理正文完全相同的多个 Chunk
        old_by_content = defaultdict(deque)

        for old_id, old_text, old_meta in zip(
            old_ids, old_texts, old_metadatas
        ):
            if not isinstance(old_text, str):
                raise RuntimeError("旧 Chunk 正文缺失，停止更新")

            old_by_content[old_text].append(
                (old_id, old_meta or {})
            )

        old_id_set = set(old_ids)
        assigned_ids = set()

        new_docs = []
        metadata_updates = []

        # 4. 逐个比较新 Chunk
        for doc in knowledge_docs:

            matches = old_by_content.get(doc.page_content)

            if matches:
                # 正文相同：直接复用原来的 Chunk ID 和向量
                chunk_id, old_meta = matches.popleft()

                # 保留创建时间，避免每次上传都改变 create_time
                if "create_time" in old_meta:
                    doc.metadata["create_time"] = old_meta["create_time"]

                doc.metadata["chunk_id"] = chunk_id

                # 若仅 chunk_index / chunk_total 等元数据变化，
                # 只更新 Metadata，不做 Embedding
                if old_meta != doc.metadata:
                    metadata_updates.append(
                        (chunk_id, doc.metadata.copy())
                    )

            else:
                # 正文不存在：需要新 Embedding
                raw = (
                    f"{doc.metadata['doc_id']}:"
                    f"{doc.metadata['chunk_index']}:"
                    f"{doc.page_content}"
                )

                chunk_id = hashlib.sha256(
                    raw.encode("utf-8")
                ).hexdigest()

                # 防止重复正文或位置变化导致新 ID 与旧 ID 冲突
                salt = 0
                while (
                    chunk_id in old_id_set
                    or chunk_id in assigned_ids
                ):
                    salt += 1
                    chunk_id = hashlib.sha256(
                        f"{raw}:{salt}".encode("utf-8")
                    ).hexdigest()

                doc.metadata["chunk_id"] = chunk_id
                new_docs.append(doc)

            assigned_ids.add(chunk_id)

        # 5. 找出新文档已经不再使用的旧记录
        stale_ids = sorted(old_id_set - assigned_ids)

        if not new_docs and not metadata_updates and not stale_ids:
            return (
                f"[跳过] {filename} 内容未变化，"
                f"复用 {len(knowledge_docs)} 个 Chunk"
            )

        batch_size = 20

        # 6. 只对新增或变化的 Chunk 做 Embedding
        for start in range(0, len(new_docs), batch_size):
            batch = new_docs[start:start + batch_size]

            self.chroma.add_texts(
                texts=[doc.page_content for doc in batch],
                metadatas=[doc.metadata for doc in batch],
                ids=[doc.metadata["chunk_id"] for doc in batch],
            )

        # 7. 仅更新 Metadata，不传 documents 或 embeddings
        # _collection 是 LangChain Chroma 封装的底层 Collection
        collection = self.chroma._collection

        for start in range(0, len(metadata_updates), batch_size):
            batch = metadata_updates[start:start + batch_size]

            collection.update(
                ids=[item[0] for item in batch],
                metadatas=[item[1] for item in batch],
            )

        # 8. 新数据写入完成后，再清理失效 Chunk
        for start in range(0, len(stale_ids), batch_size):
            batch = stale_ids[start:start + batch_size]
            self.chroma.delete(ids=batch)

        reused_count = len(knowledge_docs) - len(new_docs)

        return (
            f"[成功] {filename}\n"
            f"总 Chunk：{len(knowledge_docs)}\n"
            f"复用旧向量：{reused_count}\n"
            f"新增 Embedding：{len(new_docs)}\n"
            f"仅更新 Metadata：{len(metadata_updates)}\n"
            f"删除失效 Chunk：{len(stale_ids)}"
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
                k=20,
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
            weights=[0.5, 0.5],    # 检索分数公式：weight / (rank + c)
            id_key="chunk_id",
        )
        print(
            f"混合检索器已构建: 向量 k=20, BM25 k=20, 权重={self.hybrid_retriever.weights}")

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
                metadata={**doc.metadata,
                          "rerank_score": float(item["relevance_score"])}
            ))

        return ranked[:top_n]


def main():
    print("")


if __name__ == "__main__":
    main()
