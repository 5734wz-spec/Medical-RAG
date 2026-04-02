#!/usr/bin/env python3
"""
详细测试 RAG 系统的各个组件
"""
import sys
import os
import json
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LLM_CONFIG, EMBEDDING_CONFIG, VECTOR_DB_CONFIG
from embedding_model import EmbeddingFactory
from vector_store import FAISSVectorStore
from llm_client import LLMFactory
from retriever import MedicalRetriever
from prompt_builder import MedicalPromptBuilder


def test_embedding_model():
    """测试嵌入模型"""
    print("=" * 60)
    print("测试嵌入模型...")
    print("=" * 60)
    
    try:
        embedding_model = EmbeddingFactory.create_embedding(EMBEDDING_CONFIG)
        print("✓ 嵌入模型创建成功")
        
        # 测试嵌入
        test_texts = ["心脏病的症状", "高血压的治疗方法", "糖尿病的预防"]
        for text in test_texts:
            embedding = embedding_model.embed_query(text)
            print(f"✓ 文本 '{text}' 嵌入成功，维度: {len(embedding)}")
            print(f"  前5个值: {embedding[:5]}")
        
        return True
    except Exception as e:
        print(f"✗ 嵌入模型测试失败: {e}")
        return False


def test_vector_store():
    """测试向量数据库"""
    print("\n" + "=" * 60)
    print("测试向量数据库...")
    print("=" * 60)
    
    try:
        # 初始化向量存储
        vector_store = FAISSVectorStore(**VECTOR_DB_CONFIG)
        print(f"✓ 向量数据库加载成功")
        print(f"  统计信息: {vector_store.get_stats()}")
        
        # 测试检索
        test_queries = ["心脏病", "高血压", "糖尿病", "百日咳", "肺泡蛋白质沉积症"]
        for query in test_queries:
            results = vector_store.similarity_search(query, top_k=3)
            print(f"\n查询: '{query}'")
            print(f"  检索到 {len(results)} 个结果")
            for i, result in enumerate(results):
                doc = result['document'][:100] + "..." if len(result['document']) > 100 else result['document']
                print(f"  结果 {i+1}: {doc}")
                print(f"  相似度: {result['score']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 向量数据库测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_client():
    """测试大模型客户端"""
    print("\n" + "=" * 60)
    print("测试大模型客户端...")
    print("=" * 60)
    
    try:
        # 创建大模型客户端
        llm_client = LLMFactory.create_client(LLM_CONFIG)
        print("✓ 大模型客户端创建成功")
        
        # 测试大模型调用
        test_prompts = [
            "你好，请简单介绍一下自己。",
            "请解释什么是心脏病。",
            "如何预防高血压？"
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n测试 {i+1}: {prompt}")
            start_time = time.time()
            response = llm_client.chat(prompt)
            end_time = time.time()
            
            print(f"✓ 大模型响应成功，耗时: {end_time - start_time:.2f} 秒")
            print(f"  响应内容: {response}")
        
        return True
    except Exception as e:
        print(f"✗ 大模型客户端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_rag():
    """测试完整 RAG 流程"""
    print("\n" + "=" * 60)
    print("测试完整 RAG 流程...")
    print("=" * 60)
    
    try:
        # 初始化组件
        embedding_model = EmbeddingFactory.create_embedding(EMBEDDING_CONFIG)
        vector_store = FAISSVectorStore(**VECTOR_DB_CONFIG)
        retriever = MedicalRetriever(embedding_model, vector_store)
        prompt_builder = MedicalPromptBuilder()
        llm_client = LLMFactory.create_client(LLM_CONFIG)
        
        print("✓ 所有组件初始化成功")
        
        # 测试 RAG 流程
        test_questions = [
            "心脏病有什么症状？",
            "高血压的治疗方法是什么？",
            "百日咳的预防措施有哪些？"
        ]
        
        for question in test_questions:
            print(f"\n用户问题: {question}")
            
            # 1. 检索相关文档
            start_time = time.time()
            retrieved_docs = retriever.retrieve(question, top_k=3)
            retrieve_time = time.time() - start_time
            
            print(f"✓ 检索完成，耗时: {retrieve_time:.2f} 秒")
            print(f"  检索到 {len(retrieved_docs)} 个相关文档")
            
            # 显示检索到的文档内容
            for i, doc in enumerate(retrieved_docs):
                doc_content = doc['document'] if isinstance(doc, dict) else str(doc)
                print(f"  文档 {i+1}: {doc_content[:150]}...")
            
            # 2. 构建 Prompt
            prompt = prompt_builder.build_prompt(question, retrieved_docs)
            print(f"✓ Prompt 构建完成")
            print(f"  Prompt 长度: {len(prompt)} 字符")
            
            # 3. 调用大模型
            start_time = time.time()
            response = llm_client.chat(prompt)
            llm_time = time.time() - start_time
            
            print(f"✓ 大模型响应完成，耗时: {llm_time:.2f} 秒")
            print(f"  大模型回答: {response}")
        
        return True
    except Exception as e:
        print(f"✗ 完整 RAG 流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("RAG 系统详细测试")
    print("=" * 60)
    
    results = []
    
    # 测试各个组件
    results.append(("嵌入模型", test_embedding_model()))
    results.append(("向量数据库", test_vector_store()))
    results.append(("大模型客户端", test_llm_client()))
    results.append(("完整 RAG 流程", test_full_rag()))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！系统运行正常。")
    else:
        print("部分测试失败，请检查配置和依赖。")
    print("=" * 60)


if __name__ == "__main__":
    main()
