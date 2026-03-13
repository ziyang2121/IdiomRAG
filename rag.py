# rag.py
import os
# 强制离线模式（必须在任何相关 import 之前设置）
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
#忽略遥测警告

import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
import re
import logging
import yaml
import shutil
import gc
from typing import List, Optional, Tuple
import time
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
# 尝试导入 torch，处理 GPU 可用性
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'
    logging.warning("PyTorch 未安装，将使用 CPU 模式")

# LangChain 核心组件
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
# 暂时不用链式导入
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("rag.log"), logging.StreamHandler()]
)
logger = logging.getLogger("RAG-System")

# ==================== 配置加载 ====================
def load_config(config_path="config.yaml") -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"配置文件加载成功: {config_path}")
    return config

CONFIG = load_config()

# ==================== 嵌入模型加载（完全离线，本地路径优先） ====================
def get_embeddings() -> HuggingFaceEmbeddings:
    """根据配置获取嵌入模型实例，优先使用本地路径，完全离线"""
    emb_cfg = CONFIG['embeddings']
    model_name = emb_cfg.get('model', '')  # 仅作为标识，实际用本地路径
    model_path = emb_cfg.get('model_path')

    # 如果配置了本地路径且存在，则使用路径
    if model_path and os.path.isdir(model_path):
        logger.info(f"从本地路径加载嵌入模型: {model_path}")
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': DEVICE, 'local_files_only': True}
        )
    else:
        # 没有配置有效路径，尝试从 HuggingFace 缓存加载（必须已缓存）
        logger.info(f"未配置本地路径，尝试从缓存加载模型: {model_name}")
        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': DEVICE, 'local_files_only': True}
            )
        except Exception as e:
            logger.error(f"无法加载嵌入模型，请检查配置或模型文件。错误: {e}")
            raise

# ==================== 文档加载 ====================
def load_documents(doc_dir: str = None) -> List[Document]:
    """加载指定目录下的所有 .txt 文件"""
    if doc_dir is None:
        doc_dir = CONFIG['paths']['doc_dir']
    try:
        logger.info(f"开始加载文档 (目录: {doc_dir})")
        if not os.path.exists(doc_dir):
            os.makedirs(doc_dir)
            logger.warning(f"目录不存在，已创建: {doc_dir}")
            return []

        loader = DirectoryLoader(
            doc_dir,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True},
            show_progress=True
        )
        documents = loader.load()
        logger.info(f"成功加载 {len(documents)} 个 .txt 文件")
        return documents
    except Exception as e:
        logger.error(f"文档加载失败: {str(e)}")
        raise

# ==================== 文本清洗 ====================
def clean_text(text: str) -> str:
    """清理文本：去除多余空白、特殊字符，保留中英文及常用标点"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:()\-]', '', text)
    return text.strip()

# ==================== 语义边界分割 ====================
def split_by_semantic_boundary(documents: List[Document],
                                boundary_marker: str = "---",
                                chunk_size: int = 1000,
                                chunk_overlap: int = 100) -> List[Document]:
    """
    先按语义标记分割，再对长块二次分割，并对所有块应用文本清洗
    """
    final_chunks = []
    secondary_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for doc in documents:
        raw_blocks = doc.page_content.split(f"\n{boundary_marker}\n")
        for block in raw_blocks:
            if not block.strip():
                continue
            cleaned_block = clean_text(block)
            if len(cleaned_block) > chunk_size:
                sub_docs = secondary_splitter.create_documents([cleaned_block])
                for sub_doc in sub_docs:
                    sub_doc.metadata = doc.metadata.copy()
                final_chunks.extend(sub_docs)
            else:
                final_chunks.append(Document(
                    page_content=cleaned_block,
                    metadata=doc.metadata.copy()
                ))
    logger.info(f"文档分割完成，共生成 {len(final_chunks)} 个文本块")
    return final_chunks

# ==================== 向量数据库操作 ====================
def create_vector_db(texts: List[Document],
                     embeddings: HuggingFaceEmbeddings,
                     persist_dir: str = None,
                     force_rebuild: bool = False) -> Chroma:
    """
    创建或加载向量数据库
    - texts: 待添加的文档块（仅当库为空或force_rebuild=True时添加）
    - force_rebuild: 强制清空并重建
    """
    if persist_dir is None:
        persist_dir = CONFIG['paths']['vectorstore_dir']

    os.makedirs(persist_dir, exist_ok=True)

    if force_rebuild and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        logger.info("已清空向量库目录")

    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    try:
        existing_count = vector_db._collection.count()
    except Exception:
        existing_count = 0

    if texts and existing_count == 0:
        logger.info(f"向向量库添加 {len(texts)} 个文本块")
        vector_db.add_documents(texts)
        vector_db.persist()
        logger.info("向量库已持久化")
    elif texts and existing_count > 0:
        logger.info(f"向量库已存在 ({existing_count} 个文档)，跳过添加。如需重建请使用 /reload_db 命令")
    else:
        logger.info("无新文本添加，使用现有向量库")

    return vector_db

# ==================== LLM 加载 ====================
def load_llm(model_path: str = None) -> LlamaCpp:
    """加载 LlamaCpp 模型"""
    if model_path is None:
        model_path = CONFIG['llm']['model_path']
    try:
        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=CONFIG['llm']['n_ctx'],
            n_threads=CONFIG['llm']['n_threads'],
            verbose=False,
            temperature=CONFIG['llm']['temperature'],
            top_p=CONFIG['llm']['top_p'],
            max_tokens=CONFIG['llm']['max_tokens'],
            repeat_penalty=CONFIG['llm']['repeat_penalty'],
            stop=["<END>"]
        )
        logger.info("LLM 加载成功")
        return llm
    except Exception as e:
        logger.error(f"LLM 加载失败: {str(e)}")
        raise

# ==================== RAG 链构建（直接调用模式） ====================
def create_rag_chain(vector_db, llm):
    """创建无历史记忆的 RAG 链（直接调用模式）"""
    # 构建 prompt 模板（与原来一致）
    system_prompt = (
        "你是一个严格遵循格式的助手。你必须完全按照以下规则输出，不得有任何偏差。\n"
        "规则：\n"
        "1. 根据用户输入，从上下文中找出语义最相近的成语。\n"
        "2. 每个成语的输出格式严格为：成语：[...]，含义：[...]，出处：[...]\n"
        "3. 输出占一行，不得重复。\n"
        "4. 如果找不到任何相关成语，**必须且只能**输出：没有找到相关成语。<END>\n"
        "5. **禁止输出任何其他内容**，包括但不限于：解释、问候、感谢、代码块标记、额外空行、标点符号（除了格式中自带的）、以及“上下文”、“输出”等标签。\n"
        "6. 输出完成后，不得附加任何文字。\n\n"
        "以下是正确的输出示例：\n"
        "用户输入：他做事很细心。\n"
        "上下文：成语：[一丝不苟]，含义：[形容做事认真细致。]，出处：[清·吴敬梓《儒林外史》]\n"
        "你的输出：\n"
        "成语：[一丝不苟]，含义：[形容做事认真细致。]，出处：[清·吴敬梓《儒林外史》]\n"
        "<END>\n\n"
        "用户输入：他说话不算数。\n"
        "上下文：（无）\n"
        "你的输出：\n"
        "没有找到相关成语。<END>"
    )

    human_prompt_template = (
        "上下文：\n{context}\n\n"
        "用户输入：{input}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt_template)
    ])

    # 获取检索器
    retriever = vector_db.as_retriever(
        search_kwargs={"k": CONFIG['retrieval']['k']}
    )

    def rag_chain_func(user_input: str) -> str:
        docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])
        messages = prompt.format_messages(context=context, input=user_input)
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, 'content') else str(response)
        return raw

    logger.info("直接调用模式的 RAG 函数已创建")
    return rag_chain_func

def clean_response(text: str) -> str:
    # 1. 宽松匹配成语行：允许成分之间任意空白和标点
    # 格式：成语：[内容] 含义：[内容] 出处：[内容]
    pattern = r"成语：\[(.*?)\]\s*[，,。]?\s*含义：\[(.*?)\]\s*[，,。]?\s*出处：\[(.*?)\]"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # 去重（基于成语内容）
        seen = set()
        formatted_lines = []
        for idiom, meaning, source in matches:
            if idiom not in seen:
                seen.add(idiom)
                formatted_lines.append(f"成语：[{idiom}]，含义：[{meaning}]，出处：[{source}]")
        if formatted_lines:
            return "\n".join(formatted_lines) + "\n"
    no_idiom_patterns = [
        r"没有找到相关成语",
        r"未找到相关成语",
        r"无相关成语",
        r"找不到成语"
    ]
    # 2. 检查是否表示找不到成语（多种表述）
    for pattern in no_idiom_patterns:
        if re.search(pattern, text):
            return "没有找到相关成语。"
    # 3. 如果什么都没提取到，返回默认
    return "没有找到相关成语。"
# ==================== 完整初始化（文档加载+分割+向量库创建） ====================
def init_rag_system(embeddings: HuggingFaceEmbeddings,
                    force_rebuild_db: bool = False) -> Tuple[Chroma, object]:
    """完整初始化：加载文档、分割、创建向量库（用于首次运行或重建）"""
    logger.info("="*50)
    logger.info("执行完整初始化流程...")

    docs = load_documents()
    text_chunks = split_by_semantic_boundary(
        docs,
        boundary_marker=CONFIG['splitting']['boundary_marker'],
        chunk_size=CONFIG['splitting']['chunk_size'],
        chunk_overlap=CONFIG['splitting']['chunk_overlap']
    )
    vector_db = create_vector_db(text_chunks, embeddings, force_rebuild=force_rebuild_db)
    llm = load_llm()
    rag_chain_func = create_rag_chain(vector_db, llm)

    logger.info("完整初始化完成")
    return vector_db, rag_chain_func

# ==================== 交互主循环 ====================
def main():
    # 1. 获取嵌入模型（完全离线，本地路径优先）
    embeddings = get_embeddings()

    persist_dir = CONFIG['paths']['vectorstore_dir']
    vector_db = None
    rag_chain_func = None

    # 2. 检查现有向量库
    if os.path.exists(persist_dir):
        try:
            temp_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            count = temp_db._collection.count()
            if count > 0:
                logger.info(f"检测到现有向量库（{count} 个文档），直接加载")
                vector_db = temp_db
                llm = load_llm()
                rag_chain_func = create_rag_chain(vector_db, llm)
            else:
                logger.info("向量库为空，需要重建")
        except Exception as e:
            logger.warning(f"加载现有向量库失败，将重建: {e}")

    # 3. 如果没有成功加载，则执行完整初始化
    if vector_db is None or rag_chain_func is None:
        logger.info("未找到可用向量库，开始完整初始化...")
        vector_db, rag_chain_func = init_rag_system(embeddings, force_rebuild_db=False)

    # 4. 进入交互循环
    logger.info("\nRAG 系统已就绪！(输入 '/reload_db' 重新加载文档，'exit' 退出)")
    logger.info(f"文档目录: {CONFIG['paths']['doc_dir']}")

    while True:
        user_input = input("\n用户: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
            logger.info("系统已退出")
            break
        elif user_input == '/reload_db':
            logger.info("正在重新加载文档并重建向量库...")
            try:
                # 1. 重新加载文档并分割
                docs = load_documents()
                text_chunks = split_by_semantic_boundary(
                    docs,
                    boundary_marker=CONFIG['splitting']['boundary_marker'],
                    chunk_size=CONFIG['splitting']['chunk_size'],
                    chunk_overlap=CONFIG['splitting']['chunk_overlap']
                )

                # 2. 定义分批删除函数
                def batch_delete(collection):
                    all_ids = collection.get()['ids']
                    if not all_ids:
                        logger.info("向量库为空，无需删除")
                        return
                    total = len(all_ids)
                    batch_size = 10000  # 安全值，小于41666
                    for i in range(0, total, batch_size):
                        batch = all_ids[i:i + batch_size]
                        collection.delete(ids=batch)
                        logger.info(
                            f"已删除批次 {i // batch_size + 1}/{(total - 1) // batch_size + 1} ({len(batch)} 个文档)")
                    logger.info(f"成功删除所有 {total} 个旧文档")

                # 3. 尝试使用现有的 vector_db 对象清空
                if 'vector_db' in locals() and vector_db is not None:
                    try:
                        logger.info("清空现有向量库...")
                        batch_delete(vector_db._collection)
                        logger.info("现有向量库清空完成")
                    except Exception as e:
                        logger.warning(f"清空现有向量库失败: {e}，将尝试重建连接")
                        # 清理无效对象
                        try:
                            if hasattr(vector_db, '_client'):
                                vector_db._client.close()
                        except:
                            pass
                        del vector_db
                        vector_db = None

                # 4. 如果 vector_db 为空，则重新连接并清空
                if vector_db is None:
                    logger.info("重新连接向量库...")
                    vector_db = Chroma(
                        persist_directory=CONFIG['paths']['vectorstore_dir'],
                        embedding_function=embeddings
                    )
                    logger.info("清空新连接中的残留文档...")
                    batch_delete(vector_db._collection)

                # 5. 添加新文档
                logger.info(f"向向量库添加 {len(text_chunks)} 个新文档块...")
                vector_db.add_documents(text_chunks)
                vector_db.persist()
                logger.info("文档添加完成，向量库已更新")

                # 6. 重新创建 RAG 函数
                llm = load_llm()
                rag_chain_func = create_rag_chain(vector_db, llm)
                logger.info("向量库重建完成！")

            except Exception as e:
                logger.info(f"重建失败: {str(e)}")
                logger.error(f"重建向量库失败: {e}", exc_info=True)
            continue

        try:
            # 直接调用自定义函数，不再通过链的 invoke 获取字典
            answer = rag_chain_func(user_input)
            cleaned = clean_response(answer)
            print(f"系统: {cleaned}")
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            logger.info("处理请求时出错，请查看日志")

if __name__ == "__main__":
    main()