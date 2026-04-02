"""
构建向量数据库
将医疗数据向量化并存储到向量数据库
"""
import json
import os
from tqdm import tqdm

from config import DATA_PATH, VECTOR_DB_CONFIG, EMBEDDING_CONFIG, TEXT_SPLITTER_CONFIG
from embedding_model import EmbeddingFactory
from vector_store import FAISSVectorStore as VectorStore
from text_splitter import MedicalTextSplitter


class VectorDBBuilder:
    """向量数据库构建器"""
    
    def __init__(self):
        """初始化"""
        print("初始化向量数据库构建器...")
        
        # 初始化嵌入模型
        self.embedding_model = EmbeddingFactory.create_embedding(EMBEDDING_CONFIG)
        print(f"{self.embedding_model.__class__.__name__}初始化完成，维度: {self.embedding_model.dimensions}")
        
        # 初始化向量数据库
        self.vector_store = VectorStore(**VECTOR_DB_CONFIG)
        
        # 初始化文本切分器
        self.text_splitter = MedicalTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG['chunk_size'],
            chunk_overlap=TEXT_SPLITTER_CONFIG['chunk_overlap']
        )
        
        print("初始化完成！")
    
    def load_medical_data(self, data_path: str) -> list:
        """
        加载医疗数据
        
        Args:
            data_path: 数据文件路径
        
        Returns:
            医疗记录列表
        """
        print(f"加载医疗数据: {data_path}")
        records = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        
        print(f"共加载 {len(records)} 条医疗记录")
        return records
    
    def process_records(self, records: list) -> tuple:
        """
        处理医疗记录，切分并准备向量化
        
        Args:
            records: 医疗记录列表
        
        Returns:
            (文档列表, 元数据列表)
        """
        print("处理医疗记录...")
        
        all_documents = []
        all_metadatas = []
        
        for record in tqdm(records, desc="切分文档"):
            # 切分医疗记录
            chunks = self.text_splitter.split_medical_record(record)
            
            for chunk in chunks:
                all_documents.append(chunk['content'])
                all_metadatas.append(chunk['metadata'])
        
        print(f"共生成 {len(all_documents)} 个文档块")
        return all_documents, all_metadatas
    
    def build_vector_db(self, data_path: str = None, batch_size: int = 100):
        """
        构建向量数据库
        
        Args:
            data_path: 数据文件路径，默认使用config中的路径
            batch_size: 批处理大小
        """
        if data_path is None:
            data_path = DATA_PATH
        
        # 检查数据文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 加载数据
        records = self.load_medical_data(data_path)
        
        # 处理数据
        documents, metadatas = self.process_records(records)
        
        # 生成文档ID
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # 分批向量化并存储
        print("开始向量化文档...")
        for i in tqdm(range(0, len(documents), batch_size), desc="向量化"):
            end_idx = min(i + batch_size, len(documents))
            
            # 获取当前批次的文档
            batch_docs = documents[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            # 向量化
            batch_embeddings = self.embedding_model.embed_documents(batch_docs)
            
            # 存储到向量数据库
            self.vector_store.add_documents(
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        # 显示统计信息
        stats = self.vector_store.get_collection_stats()
        print(f"\n向量数据库构建完成！")
        print(f"统计信息: {stats}")
    
    def update_vector_db(self, data_path: str = None, batch_size: int = 100):
        """
        更新向量数据库（增量更新）
        
        Args:
            data_path: 数据文件路径
            batch_size: 批处理大小
        """
        # 获取现有统计
        stats = self.vector_store.get_collection_stats()
        print(f"当前向量数据库包含 {stats['total_documents']} 个文档")
        
        # 重新构建（简单实现）
        # 实际应用中可以实现增量更新逻辑
        print("开始更新向量数据库...")
        self.build_vector_db(data_path, batch_size)
    
    def clear_vector_db(self):
        """清空向量数据库"""
        print("清空向量数据库...")
        self.vector_store.clear_collection()
        print("向量数据库已清空")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='构建医疗知识向量数据库')
    parser.add_argument('--data-path', type=str, default=None,
                       help='医疗数据文件路径')
    parser.add_argument('--clear', action='store_true',
                       help='清空现有向量数据库')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='批处理大小')
    
    args = parser.parse_args()
    
    # 创建构建器
    builder = VectorDBBuilder()
    
    # 清空数据库（如果需要）
    if args.clear:
        builder.clear_vector_db()
    
    # 构建向量数据库
    builder.build_vector_db(
        data_path=args.data_path,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
