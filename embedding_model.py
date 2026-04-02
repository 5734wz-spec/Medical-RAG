"""
文本嵌入模型模块
支持多种嵌入模型：OpenAI、BGE、M3E等
"""
import numpy as np
from typing import List, Union
import os


class EmbeddingModel:
    """嵌入模型基类"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档列表嵌入为向量"""
        raise NotImplementedError
    
    def embed_query(self, text: str) -> List[float]:
        """将查询文本嵌入为向量"""
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI嵌入模型"""
    
    def __init__(self, api_key: str, api_base: str, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
        
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding


class BGEEmbedding(EmbeddingModel):
    """BGE嵌入模型（推荐，中文效果好）"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", device: str = "cpu", 
                 normalize_embeddings: bool = True):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装sentence-transformers库: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize_embeddings = normalize_embeddings
        print(f"BGE模型加载完成: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        embedding = self.model.encode(
            [text], 
            normalize_embeddings=self.normalize_embeddings
        )
        return embedding[0].tolist()


class M3EEmbedding(EmbeddingModel):
    """M3E嵌入模型"""
    
    def __init__(self, model_name: str = "moka-ai/m3e-base", device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装sentence-transformers库: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name, device=device)
        print(f"M3E模型加载完成: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        embedding = self.model.encode([text])
        return embedding[0].tolist()


class LocalEmbedding(EmbeddingModel):
    """本地嵌入模型（无需网络下载）"""
    
    def __init__(self, dimensions: int = 768, **kwargs):
        """
        初始化本地嵌入模型
        
        Args:
            dimensions: 嵌入维度
        """
        self.dimensions = dimensions
        print(f"本地嵌入模型初始化完成，维度: {dimensions}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        embeddings = []
        for text in texts:
            # 简单的基于字符频率的嵌入实现
            embedding = [0.0] * self.dimensions
            for i, char in enumerate(text[:self.dimensions]):
                embedding[i] = ord(char) / 256.0
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        embedding = [0.0] * self.dimensions
        for i, char in enumerate(text[:self.dimensions]):
            embedding[i] = ord(char) / 256.0
        return embedding


class DoubaoEmbedding(EmbeddingModel):
    """字节跳动Doubao嵌入模型（有免费额度）"""
    
    def __init__(self, api_key: str, api_base: str, model: str, dimensions: int = 768):
        """
        初始化Doubao嵌入模型
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
            dimensions: 嵌入维度
        """
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.dimensions = dimensions
        print(f"Doubao嵌入模型初始化完成，模型: {model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        import requests
        import json
        
        # 使用 multimodal 嵌入API端点
        url = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建 multimodal 格式的输入
        input_items = []
        for text in texts:
            input_items.append({
                "type": "text",
                "text": text
            })
        
        data = {
            "model": self.model,
            "input": input_items
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            # 打印响应内容以便调试
            print(f"API响应状态码: {response.status_code}")
            print(f"API响应内容: {response.text[:500]}...")  # 只打印前500字符
            
            result = response.json()
            print(f"解析后的响应类型: {type(result)}")
            
            if isinstance(result, dict) and 'data' in result:
                # 提取嵌入向量
                if isinstance(result['data'], list):
                    # 标准格式: data 是列表
                    embeddings = []
                    for item in result['data']:
                        if isinstance(item, dict) and 'embedding' in item:
                            embeddings.append(item['embedding'])
                        else:
                            print(f"无效的嵌入项: {item}")
                            embeddings.append([0.0] * self.dimensions)
                    return embeddings
                elif isinstance(result['data'], dict) and 'embedding' in result['data']:
                    # Multimodal 格式: data 是字典，直接包含 embedding
                    print("使用 Multimodal 格式解析嵌入")
                    embeddings = [result['data']['embedding']]
                    # 如果输入多个文本，重复使用同一个嵌入
                    if len(texts) > 1:
                        embeddings = embeddings * len(texts)
                    return embeddings
                else:
                    print(f"data 字段格式错误: {result['data']}")
                    # 使用默认嵌入
                    return [[0.0] * self.dimensions for _ in texts]
            else:
                print(f"嵌入API响应格式错误: {result}")
                # 使用默认嵌入
                return [[0.0] * self.dimensions for _ in texts]
        except Exception as e:
            print(f"嵌入API调用失败: {e}")
            import traceback
            traceback.print_exc()
            # 使用默认嵌入
            return [[0.0] * self.dimensions for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        result = self.embed_documents([text])
        if result and len(result) > 0:
            return result[0]
        else:
            return [0.0] * self.dimensions


class EmbeddingFactory:
    """嵌入模型工厂"""
    
    @staticmethod
    def create_embedding(config: dict) -> EmbeddingModel:
        """
        根据配置创建嵌入模型
        
        Args:
            config: 配置字典，来自config.py中的EMBEDDING_CONFIG
        
        Returns:
            EmbeddingModel实例
        """
        model_type = config.get('model_type', 'bge')
        
        if model_type == 'openai':
            openai_config = config.get('openai', {})
            api_key = openai_config.get('api_key')
            if not api_key:
                raise ValueError("OpenAI API密钥未设置，请设置OPENAI_API_KEY环境变量")
            
            return OpenAIEmbedding(
                api_key=api_key,
                api_base=openai_config.get('api_base', 'https://api.openai.com/v1'),
                model=openai_config.get('model', 'text-embedding-3-small')
            )
        
        elif model_type == 'bge':
            bge_config = config.get('bge', {})
            return BGEEmbedding(
                model_name=bge_config.get('model_name', 'BAAI/bge-large-zh-v1.5'),
                device=bge_config.get('device', 'cpu'),
                normalize_embeddings=bge_config.get('normalize_embeddings', True)
            )
        
        elif model_type == 'm3e':
            m3e_config = config.get('m3e', {})
            return M3EEmbedding(
                model_name=m3e_config.get('model_name', 'moka-ai/m3e-base'),
                device=m3e_config.get('device', 'cpu')
            )
        
        elif model_type == 'local':
            local_config = config.get('local', {})
            return LocalEmbedding(
                dimensions=local_config.get('dimensions', 768)
            )
        
        elif model_type == 'doubao':
            doubao_config = config.get('doubao', {})
            api_key = doubao_config.get('api_key')
            if not api_key:
                raise ValueError("Doubao API密钥未设置，请设置ARK_API_KEY环境变量")
            
            return DoubaoEmbedding(
                api_key=api_key,
                api_base=doubao_config.get('api_base', 'https://ark.cn-beijing.volces.com/api/v3'),
                model=doubao_config.get('model', 'ep-20260331103615-8nx9g'),
                dimensions=doubao_config.get('dimensions', 768)
            )
        
        else:
            raise ValueError(f"不支持的嵌入模型类型: {model_type}")


if __name__ == '__main__':
    # 测试代码
    from config import EMBEDDING_CONFIG
    
    # 创建嵌入模型
    embedding_model = EmbeddingFactory.create_embedding(EMBEDDING_CONFIG)
    
    # 测试嵌入
    texts = ["这是一个测试文本", "这是另一个测试文本"]
    embeddings = embedding_model.embed_documents(texts)
    print(f"嵌入维度: {len(embeddings[0])}")
    print(f"嵌入结果: {embeddings[0][:5]}...")  # 只显示前5个值
