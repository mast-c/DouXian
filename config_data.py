md5_path="./md5.txt"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# spliter
chunk_size = 1000
chunk_overlap = 200
separators = [
    "\n\n",                # 1. 优先保留完整段落
    "\n",                  # 2. 换行
    "。", "！", "？", "；",  # 3. 中文句子
    ". ", "! ", "? ", "; ", # 4. 英文句子
    "，", ",", "、",        # 5. 句子内部停顿
    " ",                   # 6. 空格
    ""                     # 7. 最后按字符兜底
]
max_split_char_number=120  #文本分割的阈值

