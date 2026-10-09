md5_path="./md5.txt"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# spliter
chunk_size = 400
chunk_overlap = 50
separators = ["\n", "\n\n", "。", "！", "？", ",", "!", "?", "|", ".", ";"]
max_split_char_number=120  #文本分割的阈值


#
similarity_threshold = 2        # 检索返回匹配的文档数量
session_config={
        "configurable":{
            "session_id":"user_001"
        }
    }