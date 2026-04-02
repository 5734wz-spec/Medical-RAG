"""
重排序模块
对检索结果进行精排，提升相关性
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Reranker:
    """重排序器基类"""
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对文档进行重排序"""
        raise NotImplementedError


class TFIDFReranker(Reranker):
    """基于TF-IDF的重排序器"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用TF-IDF重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
        
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        # 提取文档内容
        doc_texts = [doc['document'] for doc in documents]
        
        # 构建TF-IDF矩阵
        all_texts = [query] + doc_texts
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # 计算查询与文档的相似度
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # 更新分数并重排序
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(similarities[i])
            
            # 按重排序分数排序
            documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
        except Exception as e:
            print(f"TF-IDF重排序失败: {e}")
        
        return documents


class CrossEncoderReranker(Reranker):
    """基于交叉编码器的重排序器（效果更好，但需要更多计算）"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化
        
        Args:
            model_name: 交叉编码器模型名称
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("请安装sentence-transformers库: pip install sentence-transformers")
        
        self.model = CrossEncoder(model_name)
        print(f"交叉编码器重排序器加载完成: {model_name}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用交叉编码器重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
        
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        # 构建查询-文档对
        pairs = [[query, doc['document']] for doc in documents]
        
        # 预测相关性分数
        scores = self.model.predict(pairs)
        
        # 更新分数
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
        
        # 按重排序分数排序
        documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return documents


class LengthPenaltyReranker(Reranker):
    """基于长度惩罚的重排序器"""
    
    def __init__(self, base_score_weight: float = 0.7, length_weight: float = 0.3,
                 ideal_length: int = 300):
        """
        初始化
        
        Args:
            base_score_weight: 基础分数权重
            length_weight: 长度分数权重
            ideal_length: 理想文档长度
        """
        self.base_score_weight = base_score_weight
        self.length_weight = length_weight
        self.ideal_length = ideal_length
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用长度惩罚重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
        
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        for doc in documents:
            # 基础分数（原始相似度）
            base_score = doc.get('score', 0.5)
            
            # 长度分数（越接近理想长度越好）
            doc_length = len(doc['document'])
            length_score = 1.0 - abs(doc_length - self.ideal_length) / self.ideal_length
            length_score = max(0, length_score)  # 确保非负
            
            # 综合分数
            final_score = (self.base_score_weight * base_score + 
                          self.length_weight * length_score)
            
            doc['rerank_score'] = final_score
        
        # 按综合分数排序
        documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return documents


class MedicalReranker(Reranker):
    """医疗专用重排序器"""
    
    def __init__(self, use_cross_encoder: bool = False):
        """
        初始化
        
        Args:
            use_cross_encoder: 是否使用交叉编码器
        """
        self.use_cross_encoder = use_cross_encoder
        
        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker()
        else:
            self.tfidf_reranker = TFIDFReranker()
        
        self.length_reranker = LengthPenaltyReranker()
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               intent: str = None) -> List[Dict[str, Any]]:
        """
        医疗文档重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            intent: 查询意图
        
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        # 第一步：使用语义重排序
        if self.use_cross_encoder:
            documents = self.cross_encoder.rerank(query, documents)
        else:
            documents = self.tfidf_reranker.rerank(query, documents)
        
        # 第二步：应用长度惩罚
        documents = self.length_reranker.rerank(query, documents)
        
        # 第三步：根据意图调整
        if intent:
            documents = self._adjust_by_intent(documents, intent)
        
        return documents
    
    def _adjust_by_intent(self, documents: List[Dict[str, Any]], 
                         intent: str) -> List[Dict[str, Any]]:
        """
        根据意图调整排序
        
        Args:
            documents: 文档列表
            intent: 查询意图
        
        Returns:
            调整后的文档列表
        """
        # 意图到字段的映射
        intent_field_map = {
            'disease_symptom': ['symptom'],
            'disease_cause': ['cause'],
            'disease_prevent': ['prevent'],
            'disease_cureway': ['cure_way'],
            'disease_cureprob': ['cured_prob'],
            'disease_lasttime': ['cure_lasttime'],
            'disease_easyget': ['easy_get'],
            'disease_acompany': ['acompany'],
            'disease_not_food': ['not_eat'],
            'disease_do_food': ['do_eat', 'recommand_eat'],
            'disease_drug': ['common_drug', 'recommand_drug'],
            'disease_check': ['check'],
        }
        
        # 获取相关字段
        relevant_fields = intent_field_map.get(intent, [])
        
        if not relevant_fields:
            return documents
        
        # 提升相关字段文档的分数
        for doc in documents:
            metadata = doc.get('metadata', {})
            doc_field = metadata.get('field', '')
            
            # 如果文档字段与意图相关，提升分数
            if doc_field in relevant_fields:
                boost = 0.2  # 提升幅度
                doc['rerank_score'] = doc.get('rerank_score', 0) + boost
                doc['intent_match'] = True
            else:
                doc['intent_match'] = False
        
        # 重新排序
        documents.sort(key=lambda x: (x.get('intent_match', False), 
                                     x.get('rerank_score', 0)), 
                      reverse=True)
        
        return documents


class SimpleReranker(Reranker):
    """简单重排序器（基于规则）"""
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        简单重排序
        
        规则：
        1. 包含查询关键词的文档优先
        2. 文档长度适中的优先
        """
        if not documents:
            return []
        
        query_keywords = set(query.lower().split())
        
        for doc in documents:
            doc_text = doc['document'].lower()
            
            # 关键词匹配分数
            keyword_matches = sum(1 for kw in query_keywords if kw in doc_text)
            keyword_score = keyword_matches / len(query_keywords) if query_keywords else 0
            
            # 基础分数
            base_score = doc.get('score', 0.5)
            
            # 综合分数
            final_score = 0.6 * base_score + 0.4 * keyword_score
            
            doc['rerank_score'] = final_score
            doc['keyword_matches'] = keyword_matches
        
        # 排序
        documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return documents


def get_reranker(reranker_type: str = "simple", **kwargs) -> Reranker:
    """
    获取重排序器实例
    
    Args:
        reranker_type: 重排序器类型
        **kwargs: 其他参数
    
    Returns:
        重排序器实例
    """
    if reranker_type == "simple":
        return SimpleReranker()
    elif reranker_type == "tfidf":
        return TFIDFReranker()
    elif reranker_type == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    elif reranker_type == "length":
        return LengthPenaltyReranker(**kwargs)
    elif reranker_type == "medical":
        return MedicalReranker(**kwargs)
    else:
        raise ValueError(f"未知的重排序器类型: {reranker_type}")


if __name__ == '__main__':
    # 测试代码
    test_documents = [
        {
            'id': '1',
            'document': '高血压的症状包括头痛、头晕、心悸等。患者可能会感到胸闷、气短。',
            'metadata': {'field': 'symptom'},
            'score': 0.85
        },
        {
            'id': '2',
            'document': '高血压是一种常见的慢性病，需要长期治疗。',
            'metadata': {'field': 'desc'},
            'score': 0.80
        },
        {
            'id': '3',
            'document': '预防高血压需要控制体重、减少盐摄入、规律运动。',
            'metadata': {'field': 'prevent'},
            'score': 0.75
        }
    ]
    
    query = "高血压的症状有哪些？"
    
    # 测试简单重排序
    print("=== 简单重排序 ===")
    reranker = SimpleReranker()
    results = reranker.rerank(query, test_documents.copy())
    for i, doc in enumerate(results):
        print(f"{i+1}. 分数: {doc['rerank_score']:.4f}, 内容: {doc['document'][:30]}...")
    
    # 测试TF-IDF重排序
    print("\n=== TF-IDF重排序 ===")
    reranker = TFIDFReranker()
    results = reranker.rerank(query, test_documents.copy())
    for i, doc in enumerate(results):
        print(f"{i+1}. 分数: {doc['rerank_score']:.4f}, 内容: {doc['document'][:30]}...")
    
    # 测试医疗重排序
    print("\n=== 医疗重排序（带意图） ===")
    reranker = MedicalReranker(use_cross_encoder=False)
    results = reranker.rerank(query, test_documents.copy(), intent='disease_symptom')
    for i, doc in enumerate(results):
        print(f"{i+1}. 分数: {doc['rerank_score']:.4f}, "
              f"意图匹配: {doc.get('intent_match', False)}, "
              f"内容: {doc['document'][:30]}...")
