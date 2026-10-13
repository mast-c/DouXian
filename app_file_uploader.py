import streamlit as st
from knowledge_base import KnowledgeBaseService
from pathlib import Path
import pypdf  # 解析 PDF 的库 (需安装: pip install pypdf)
import docx    # 解析 Word 的库 (需安装: pip install python-docx)

# ---------------- 辅助函数：多格式文件解析 ----------------
def parse_file_content(uploaded_file):
    """根据文件后缀名，智能提取文本内容"""
    file_name = uploaded_file.name.lower()
    
    try:
        if file_name.endswith('.txt') or file_name.endswith('.md'):
            return uploaded_file.getvalue().decode("utf-8")
            
        elif file_name.endswith('.pdf'):
            reader = pypdf.PdfReader(uploaded_file)
            return "\n\n".join([page.extract_text() for page in reader.pages])
            
        elif file_name.endswith('.docx'):
            # docx 库需要传入字节流
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
            
        else:
            return None, f"不支持的文件格式: {file_name}"
    except Exception as e:
        return None, f"文件解析失败，原因: {str(e)}"

# ---------------- 页面 UI 渲染 ----------------

# 1. 文件上传组件 (支持 txt, pdf, docx, md)
uploader_file = st.file_uploader(
    "请上传知识库文档 (支持 TXT / PDF / DOCX / MD)",
    type=['txt', 'pdf', 'docx', 'md'],
    accept_multiple_files=False, 
)

# 2. 初始化知识库服务
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

# 3. 处理上传文件
if uploader_file is not None:
    # 提取文件基础信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024  # KB

    # 展示文件信息
    st.subheader(f"文件: {file_name}")
    st.write(f"格式: {file_type} | 大小: {file_size:.2f} KB")

    # 进度提示
    with st.spinner("正在解析并录入知识库，请稍候..."):
        # 调用解析函数
        content_or_error = parse_file_content(uploader_file)
        
        # 判断是解析成功（返回字符串）还是失败（返回元组）
        if isinstance(content_or_error, tuple):
            st.error(content_or_error[1])
        else:
            text_content = content_or_error
            
            # 展示前 200 个字符供用户预览
            with st.expander("预览提取到的文本内容"):
                st.code(text_content[:500] + ("..." if len(text_content) > 500 else ""))

            # 录入知识库
            res = st.session_state["service"].upload_by_str(text_content, file_name)
            st.success(f"成功: {res}")
