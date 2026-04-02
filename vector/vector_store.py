"""
向量数据库操作模块
基于 ChromaDB 实现向量存储和检索
"""
import os
from typing import List, Dict, Any, Optional
import json


class VectorStore:
    """向量数据库存储类"""
    
    def __init__(self, persist_directory: str, collection_name: str = "medical_knowledge"):
        """
        初始化向量数据库
        
        Args:
            persist_directory: 数据持久化目录
            collection_name: 集合名称
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("请安装chromadb库: pip install chromadb")
        
        self.persist_directory = persist_directory
        
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化ChromaDB客户端（使用新的API）
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"向量数据库初始化完成: {persist_directory}/{collection_name}")
    
    def add_documents(self, 
                      documents: List[str], 
                      embeddings: List[List[float]], 
                      metadatas: Optional[List[Dict[str, Any]]] = None,
                      ids: Optional[List[str]] = None):
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档内容列表
            embeddings: 文档嵌入向量列表
            metadatas: 文档元数据列表
            ids: 文档ID列表
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]
        
        # 分批添加，避免单次请求过大
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            print(f"已添加 {end_idx}/{len(documents)} 个文档")
        
        # 持久化
        self.client.persist()
        print(f"成功添加 {len(documents)} 个文档到向量数据库")
    
    def similarity_search(self, 
                         query_embedding: List[float], 
                         top_k: int = 10,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        相似度搜索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filter_dict: 过滤条件
        
        Returns:
            搜索结果列表，每个结果包含文档内容、元数据和相似度分数
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'score': 1 - results['distances'][0][i]  # 转换为相似度分数
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name,
            'persist_directory': self.persist_directory
        }
    
    def delete_collection(self):
        """删除集合"""
        self.client.delete_collection(self.collection.name)
        print(f"集合 {self.collection.name} 已删除")
    
    def clear_collection(self):
        """清空集合中的所有文档"""
        # 获取所有文档ID
        all_docs = self.collection.get()
        if all_docs['ids']:
            self.collection.delete(ids=all_docs['ids'])
            self.client.persist()
            print(f"已清空集合，删除 {len(all_docs['ids'])} 个文档")


class FAISSVectorStore(VectorStore):
    """基于FAISS的向量存储（备选方案）"""
    
    def __init__(self, persist_directory: str, collection_name: str = "medical_knowledge"):
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("请安装faiss库: pip install faiss-cpu 或 pip install faiss-gpu")
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.index_file = os.path.join(persist_directory, f"{collection_name}.index")
        self.metadata_file = os.path.join(persist_directory, f"{collection_name}_metadata.json")
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.index = None
        self.documents = []
        self.metadatas = []
        self.dimension = None
        
        # 尝试加载已有索引
        if os.path.exists(self.index_file):
            self._load_index()
    
    def _load_index(self):
        """加载已有索引"""
        import faiss
        import numpy as np
        
        self.index = faiss.read_index(self.index_file)
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.documents = data['documents']
            self.metadatas = data['metadatas']
            self.dimension = data['dimension']
        
        print(f"已加载FAISS索引，包含 {len(self.documents)} 个文档")
    
    def _save_index(self):
        """保存索引"""
        import faiss
        
        faiss.write_index(self.index, self.index_file)
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'dimension': self.dimension
            }, f, ensure_ascii=False, indent=2)
    
    def add_documents(self, 
                      documents: List[str], 
                      embeddings: List[List[float]], 
                      metadatas: Optional[List[Dict[str, Any]]] = None,
                      ids: Optional[List[str]] = None):
        """添加文档"""
        import faiss
        import numpy as np
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]
        
        # 检查嵌入维度
        new_dimension = len(embeddings[0])
        
        # 初始化索引或重建索引（如果维度不匹配）
        if self.index is None:
            # 新索引
            self.dimension = new_dimension
            self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度
            print(f"创建新的FAISS索引，维度: {self.dimension}")
        elif self.dimension != new_dimension:
            # 维度不匹配，重建索引
            print(f"维度不匹配: 现有索引维度 {self.dimension}, 新嵌入维度 {new_dimension}")
            print("重建FAISS索引...")
            self.clear_collection()
            self.dimension = new_dimension
            self.index = faiss.IndexFlatIP(self.dimension)
            print(f"创建新的FAISS索引，维度: {self.dimension}")
        
        # 添加向量
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        
        # 保存文档和元数据
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        # 保存索引
        self._save_index()
        print(f"成功添加 {len(documents)} 个文档到FAISS索引")
    
    def similarity_search(self, 
                         query_embedding: List[float], 
                         top_k: int = 10,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """相似度搜索"""
        import numpy as np
        
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_array, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            
            # 如果有过滤条件，检查是否匹配
            if filter_dict:
                match = all(
                    self.metadatas[idx].get(k) == v 
                    for k, v in filter_dict.items()
                )
                if not match:
                    continue
            
            results.append({
                'id': str(idx),
                'document': self.documents[idx],
                'metadata': self.metadatas[idx],
                'distance': float(distances[0][i]),
                'score': float(distances[0][i])
            })
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_documents': len(self.documents),
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'dimension': self.dimension
        }
    
    def clear_collection(self):
        """清空集合"""
        if self.index is not None:
            self.index.reset()
        self.documents = []
        self.metadatas = []
        
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
        
        print("FAISS索引已清空")


if __name__ == '__main__':
    # 测试代码
    from config import VECTOR_DB_CONFIG
    
    # 创建向量存储
    store = VectorStore(**VECTOR_DB_CONFIG)
    
    # 测试添加文档
    test_docs = ["这是测试文档1", "这是测试文档2", "这是测试文档3"]
    test_embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    test_metadatas = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
    
    store.add_documents(test_docs, test_embeddings, test_metadatas)
    
    # 测试搜索
    results = store.similarity_search([0.15] * 1024, top_k=2)
    print(f"搜索结果: {results}")
    
    # 查看统计
    stats = store.get_collection_stats()
    print(f"统计信息: {stats}")
