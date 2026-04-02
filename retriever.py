"""
检索器模块
实现向量检索、关键词检索和混合检索
"""
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict

from config import RETRIEVER_CONFIG
from embedding_model import EmbeddingModel
from vector_store import VectorStore


class Retriever:
    """检索器基类"""
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """检索相关文档"""
        raise NotImplementedError


class VectorRetriever(Retriever):
    """向量检索器"""
    
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        """
        初始化
        
        Args:
            embedding_model: 嵌入模型
            vector_store: 向量数据库
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = None, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        向量检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_dict: 过滤条件
        
        Returns:
            检索结果列表
        """
        if top_k is None:
            top_k = RETRIEVER_CONFIG['vector_top_k']
        
        # 将查询向量化
        query_embedding = self.embedding_model.embed_query(query)
        
        # 在向量数据库中搜索
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results


class KeywordRetriever(Retriever):
    """关键词检索器（基于倒排索引）"""
    
    def __init__(self, documents: List[str] = None, metadatas: List[Dict] = None):
        """
        初始化
        
        Args:
            documents: 文档列表
            metadatas: 元数据列表
        """
        self.inverted_index = defaultdict(set)
        self.documents = documents or []
        self.metadatas = metadatas or []
        
        if documents:
            self._build_index()
    
    def _build_index(self):
        """构建倒排索引"""
        for idx, doc in enumerate(self.documents):
            # 简单的分词（按字符）
            words = set(doc)
            for word in words:
                if len(word.strip()) > 0:
                    self.inverted_index[word].add(idx)
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """添加文档"""
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(documents))
        
        # 更新索引
        for i, doc in enumerate(documents):
            idx = start_idx + i
            words = set(doc)
            for word in words:
                if len(word.strip()) > 0:
                    self.inverted_index[word].add(idx)
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            检索结果列表
        """
        if top_k is None:
            top_k = RETRIEVER_CONFIG['keyword_top_k']
        
        # 计算每个文档的匹配分数
        doc_scores = defaultdict(float)
        
        # 对查询进行简单分词
        query_words = set(query)
        
        for word in query_words:
            if len(word.strip()) > 0 and word in self.inverted_index:
                for doc_idx in self.inverted_index[word]:
                    doc_scores[doc_idx] += 1
        
        # 按分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 格式化结果
        results = []
        for doc_idx, score in sorted_docs[:top_k]:
            results.append({
                'id': str(doc_idx),
                'document': self.documents[doc_idx],
                'metadata': self.metadatas[doc_idx] if doc_idx < len(self.metadatas) else {},
                'score': score,
                'distance': 1.0 / (score + 1)  # 转换为距离
            })
        
        return results


class HybridRetriever(Retriever):
    """混合检索器（向量检索 + 关键词检索）"""
    
    def __init__(self, 
                 vector_retriever: VectorRetriever,
                 keyword_retriever: Optional[KeywordRetriever] = None,
                 vector_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        """
        初始化
        
        Args:
            vector_retriever: 向量检索器
            keyword_retriever: 关键词检索器
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
        """
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            检索结果列表
        """
        if top_k is None:
            top_k = RETRIEVER_CONFIG['vector_top_k']
        
        # 向量检索
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
        
        # 关键词检索（如果有）
        keyword_results = []
        if self.keyword_retriever:
            keyword_results = self.keyword_retriever.retrieve(query, top_k=top_k)
        
        # 融合结果
        fused_results = self._fuse_results(vector_results, keyword_results)
        
        return fused_results[:top_k]
    
    def _fuse_results(self, vector_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """
        融合向量检索和关键词检索结果
        
        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
        
        Returns:
            融合后的结果
        """
        # 使用RRF (Reciprocal Rank Fusion) 算法
        k = 60  # RRF常数
        
        doc_scores = defaultdict(float)
        doc_info = {}
        
        # 处理向量检索结果
        for rank, result in enumerate(vector_results):
            doc_id = result['id']
            doc_scores[doc_id] += self.vector_weight * (1.0 / (k + rank + 1))
            if doc_id not in doc_info:
                doc_info[doc_id] = result
        
        # 处理关键词检索结果
        for rank, result in enumerate(keyword_results):
            doc_id = result['id']
            doc_scores[doc_id] += self.keyword_weight * (1.0 / (k + rank + 1))
            if doc_id not in doc_info:
                doc_info[doc_id] = result
        
        # 按融合分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 格式化结果
        results = []
        for doc_id, score in sorted_docs:
            result = doc_info[doc_id].copy()
            result['fusion_score'] = score
            results.append(result)
        
        return results


class MedicalRetriever:
    """医疗专用检索器"""
    
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        """
        初始化
        
        Args:
            embedding_model: 嵌入模型
            vector_store: 向量数据库
        """
        self.vector_retriever = VectorRetriever(embedding_model, vector_store)
        self.config = RETRIEVER_CONFIG
    
    def retrieve_by_intent(self, 
                          query: str, 
                          intent: str,
                          entities: Dict[str, List[str]] = None) -> List[Dict[str, Any]]:
        """
        基于意图的检索
        
        Args:
            query: 查询文本
            intent: 意图类型（如 disease_symptom, disease_cause 等）
            entities: 识别的实体
        
        Returns:
            检索结果
        """
        filter_dict = None
        
        # 根据意图构建过滤条件
        if entities:
            if 'disease' in entities and entities['disease']:
                # 如果识别到疾病实体，优先检索该疾病相关内容
                disease_name = entities['disease'][0]
                filter_dict = {'disease': disease_name}
        
        # 优化检索参数：增加检索结果数量
        base_top_k = self.config['vector_top_k']
        top_k = base_top_k * 3  # 增加到原来的3倍
        
        # 执行检索
        results = self.vector_retriever.retrieve(query, top_k=top_k, filter_dict=filter_dict)
        
        # 如果没有结果，尝试不加过滤条件
        if not results and filter_dict:
            results = self.vector_retriever.retrieve(query, top_k=top_k)
        
        # 过滤低相似度结果（设置相似度阈值）
        # 注意：FAISS使用内积相似度，score值越大表示越相似
        filtered_results = []
        if results:
            # 打印前几个结果的相似度分数，以便调整阈值
            print(f"前5个结果的相似度分数: {[result.get('score', 0) for result in results[:5]]}")
            
        for result in results:
            score = result.get('score', 0)
            # 降低相似度阈值，内积大于0的视为相关
            if score > 0:
                filtered_results.append(result)
        
        # 确保返回足够的结果，但不超过原始top_k
        return filtered_results[:base_top_k]
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        通用检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            检索结果
        """
        if top_k is None:
            top_k = self.config['vector_top_k']
        
        # 优化检索参数：增加检索结果数量
        expanded_top_k = top_k * 3
        
        # 执行检索
        results = self.vector_retriever.retrieve(query, top_k=expanded_top_k)
        
        # 过滤低相似度结果
        # 注意：FAISS使用内积相似度，score值越大表示越相似
        filtered_results = []
        if results:
            # 打印前几个结果的相似度分数，以便调整阈值
            print(f"前5个结果的相似度分数: {[result.get('score', 0) for result in results[:5]]}")
            
        for result in results:
            score = result.get('score', 0)
            # 降低相似度阈值，内积大于0的视为相关
            if score > 0:
                filtered_results.append(result)
        
        # 确保返回足够的结果，但不超过原始top_k
        return filtered_results[:top_k]


if __name__ == '__main__':
    # 测试代码
    from config import EMBEDDING_CONFIG, VECTOR_DB_CONFIG
    from embedding_model import EmbeddingFactory
    from vector_store import VectorStore
    
    # 初始化组件
    embedding_model = EmbeddingFactory.create_embedding(EMBEDDING_CONFIG)
    vector_store = VectorStore(**VECTOR_DB_CONFIG)
    
    # 创建检索器
    retriever = MedicalRetriever(embedding_model, vector_store)
    
    # 测试检索
    query = "高血压的症状有哪些？"
    results = retriever.retrieve(query, top_k=5)
    
    print(f"查询: {query}")
    print(f"检索到 {len(results)} 条结果:")
    for i, result in enumerate(results):
        print(f"\n结果 {i+1}:")
        print(f"  内容: {result['document'][:100]}...")
        print(f"  分数: {result['score']}")
        print(f"  元数据: {result['metadata']}")
