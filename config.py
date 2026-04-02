"""
RAG医疗问答系统配置文件
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_PATH = os.path.join(BASE_DIR, 'data/medical.json')
VECTOR_DB_PATH = os.path.join(BASE_DIR, 'vector_db')
DICT_DIR = os.path.join(BASE_DIR, 'dict')

# 向量数据库配置
VECTOR_DB_CONFIG = {
    'persist_directory': VECTOR_DB_PATH,
    'collection_name': 'medical_knowledge',
}

# 嵌入模型配置
EMBEDDING_CONFIG = {
    # 选择使用的模型: 'openai', 'bge', 'm3e', 'local', 'baidu', 'ali', 'tencent', 'doubao'
    'model_type': 'doubao',  # 使用豆包嵌入模型
    
    # OpenAI配置
    'openai': {
        'api_key': os.getenv('OPENAI_API_KEY', ''),
        'api_base': os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        'model': 'text-embedding-3-small',
        'dimensions': 1536,
    },
    
    # BGE配置 (推荐，开源中文效果好)
    'bge': {
        'model_name': 'BAAI/bge-large-zh-v1.5',
        'dimensions': 1024,
        'device': 'cpu',  # 或 'cuda' 如果有GPU
        'normalize_embeddings': True,
    },
    
    # M3E配置
    'm3e': {
        'model_name': 'moka-ai/m3e-base',
        'dimensions': 768,
        'device': 'cpu',
    },
    
    # 本地模型配置（无需网络下载）
    'local': {
        'model_name': 'local',
        'dimensions': 768,
        'device': 'cpu',
    },
    
    # 百度文心Embedding配置（有免费额度）
    'baidu': {
        'api_key': os.getenv('BAIDU_API_KEY', ''),
        'secret_key': os.getenv('BAIDU_SECRET_KEY', ''),
        'model': 'text-embedding-v1',
        'dimensions': 768,
    },
    
    # 阿里通义Embedding配置（有免费额度）
    'ali': {
        'api_key': os.getenv('ALI_API_KEY', ''),
        'model': 'text-embedding-adas',
        'dimensions': 768,
    },
    
    # 腾讯混元Embedding配置（有免费额度）
    'tencent': {
        'api_key': os.getenv('TENCENT_API_KEY', ''),
        'secret_key': os.getenv('TENCENT_SECRET_KEY', ''),
        'model': 'embedding-zh',
        'dimensions': 768,
    },
    
    # 字节跳动Doubao Embedding配置（有免费额度）
    'doubao': {
        'api_key': os.getenv('ARK_API_KEY', '5ec2acc9-c32b-4a7e-a135-21640c83dc21'),
        'api_base': 'https://ark.cn-beijing.volces.com/api/v3',
        'model': 'ep-20260401151626-9j27s',
        'dimensions': 2048,
    },
}

# 大模型配置
LLM_CONFIG = {
    # 选择使用的模型: 'openai', 'qwen', 'wenxin', 'zhipu', 'ollama', 'doubao', 'gemini'
    'model_type': 'doubao',  # 使用字节跳动Doubao API（有免费额度）
    
    # OpenAI配置
    'openai': {
        'api_key': os.getenv('OPENAI_API_KEY', ''),
        'api_base': os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'max_tokens': 2000,
        'top_p': 0.9,
    },
    
    # 通义千问配置
    'qwen': {
        'api_key': os.getenv('DASHSCOPE_API_KEY', ''),
        'api_base': 'https://dashscope.aliyuncs.com/api/v1',
        'model': 'qwen-turbo',
        'temperature': 0.7,
        'max_tokens': 2000,
    },
    
    # 文心一言配置
    'wenxin': {
        'api_key': os.getenv('WENXIN_API_KEY', ''),
        'secret_key': os.getenv('WENXIN_SECRET_KEY', ''),
        'model': 'ernie-bot-turbo',
        'temperature': 0.7,
        'max_tokens': 2000,
    },
    
    # 智谱AI配置
    'zhipu': {
        'api_key': os.getenv('ZHIPU_API_KEY', ''),
        'model': 'chatglm_turbo',
        'temperature': 0.7,
        'max_tokens': 2000,
    },
    
    # 字节跳动Doubao配置（有免费额度）
    'doubao': {
        'api_key': os.getenv('ARK_API_KEY', ''),
        'api_base': '',
        'model': '',  # 用户提供的模型ID
        'temperature': 0.7,
        'max_tokens': 2000,
    },
    
    # Google Gemini配置（有免费额度）
    'gemini': {
        'api_key': os.getenv('GEMINI_API_KEY', ''),
        'api_base': 'https://generativelanguage.googleapis.com/v1',
        'model': 'gemini-1.5-flash',
        'temperature': 0.7,
        'max_tokens': 2000,
    },
    
    # Ollama本地模型配置（完全免费）
    'ollama': {
        'model': 'llama3:8b',  # 可以替换为其他模型，如 qwen2:7b, gemma2:9b 等
        'api_base': 'http://localhost:11434',
        'temperature': 0.7,
        'max_tokens': 2000,
    },
}

# 检索配置
RETRIEVER_CONFIG = {
    # 向量检索Top-K
    'vector_top_k': 10,
    # 关键词检索Top-K
    'keyword_top_k': 5,
    # 混合检索权重 (向量权重, 关键词权重)
    'hybrid_weights': (0.7, 0.3),
    # 重排序后的Top-K
    'rerank_top_k': 5,
    # 相似度阈值
    'similarity_threshold': 0.5,
}

# 文本切分配置
TEXT_SPLITTER_CONFIG = {
    'chunk_size': 500,
    'chunk_overlap': 100,
    'separators': ['\n\n', '\n', '。', '；', '，', ' ', ''],
}

# Prompt模板配置
PROMPT_TEMPLATES = {
    'medical_qa': '''你是一个专业的医疗知识助手。请基于以下检索到的医疗知识，回答用户的问题。

【检索到的相关知识】
{context}

【用户问题】
{question}

【回答要求】
1. 请基于上述检索到的知识进行回答
2. 如果检索知识不足以回答问题，请明确说明
3. 回答要准确、专业、易懂
4. 如有多个相关信息，请分点说明
5. 不要编造不在检索知识中的内容

请回答：''',

    'disease_symptom': '''根据以下疾病信息，回答关于症状的问题。

【疾病信息】
{context}

【问题】
{question}

请列出该疾病的症状：''',

    'disease_treatment': '''根据以下疾病信息，回答关于治疗方案的问题。

【疾病信息】
{context}

【问题】
{question}

请说明治疗方案：''',
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(BASE_DIR, 'rag.log'),
}
