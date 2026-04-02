#!/usr/bin/env python3
"""
测试大模型连接和响应
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LLM_CONFIG, EMBEDDING_CONFIG, VECTOR_DB_CONFIG
from llm_client import LLMFactory
from embedding_model import EmbeddingFactory
from vector_store import FAISSVectorStore


def test_llm_connection():
    """测试大模型连接"""
    print("=" * 60)
    print("测试大模型连接...")
    print("=" * 60)
    
    try:
        # 创建大模型客户端
        client = LLMFactory.create_client(LLM_CONFIG)
        print("✓ 大模型客户端创建成功")
        
        # 测试简单对话
        test_prompt = "你好，请简单介绍一下自己。"
        print(f"\n发送测试消息: {test_prompt}")
        
        response = client.chat(test_prompt)
        print(f"\n大模型响应:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        if response and len(response.strip()) > 0:
            print("✓ 大模型连接测试通过！")
            return True
        else:
            print("✗ 大模型返回空响应")
            return False
            
    except Exception as e:
        print(f"✗ 大模型连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding():
    """测试嵌入模型"""
    print("\n" + "=" * 60)
    print("测试嵌入模型...")
    print("=" * 60)
    
    try:
        # 创建嵌入模型
        embedding_model = EmbeddingFactory.create_embedding(EMBEDDING_CONFIG)
        print("✓ 嵌入模型创建成功")
        
        # 测试文本嵌入
        test_text = "心脏病的症状包括胸痛、气短等。"
        print(f"\n测试文本: {test_text}")
        
        embedding = embedding_model.embed_query(test_text)
        print(f"✓ 嵌入向量维度: {len(embedding)}")
        print(f"✓ 嵌入向量前5个值: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ 嵌入模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_search():
    """测试向量检索"""
    print("\n" + "=" * 60)
    print("测试向量检索...")
    print("=" * 60)
    
    try:
        # 创建嵌入模型
        embedding_model = EmbeddingFactory.create_embedding(EMBEDDING_CONFIG)
        
        # 创建向量存储
        vector_store = FAISSVectorStore(**VECTOR_DB_CONFIG)
        
        # 获取统计信息
        stats = vector_store.get_collection_stats()
        print(f"✓ 向量数据库统计: {stats}")
        
        # 测试检索
        query = "心脏病"
        print(f"\n测试查询: {query}")
        
        query_embedding = embedding_model.embed_query(query)
        results = vector_store.similarity_search(query_embedding, top_k=3)
        
        print(f"✓ 检索到 {len(results)} 个结果")
        
        for i, result in enumerate(results):
            print(f"\n结果 {i+1}:")
            print(f"  文档: {result['document'][:100]}...")
            print(f"  相似度: {result['score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 向量检索测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """测试完整流程"""
    print("\n" + "=" * 60)
    print("测试完整RAG流程...")
    print("=" * 60)
    
    try:
        # 初始化组件
        embedding_model = EmbeddingFactory.create_embedding(EMBEDDING_CONFIG)
        vector_store = FAISSVectorStore(**VECTOR_DB_CONFIG)
        llm_client = LLMFactory.create_client(LLM_CONFIG)
        
        print("✓ 所有组件初始化成功")
        
        # 测试查询
        query = "心脏病有什么症状？"
        print(f"\n用户问题: {query}")
        
        # 检索
        query_embedding = embedding_model.embed_query(query)
        results = vector_store.similarity_search(query_embedding, top_k=3)
        print(f"✓ 检索到 {len(results)} 个相关文档")
        
        if not results:
            print("✗ 未检索到任何文档，无法生成回答")
            return False
        
        # 构建Prompt
        context = "\n\n".join([r['document'] for r in results])
        prompt = f"""基于以下医疗知识，回答用户的问题。

【相关知识】
{context}

【用户问题】
{query}

请回答："""
        
        print("\n构建的Prompt:")
        print("-" * 60)
        print(prompt[:300] + "...")
        print("-" * 60)
        
        # 调用大模型
        print("\n正在调用大模型...")
        response = llm_client.chat(prompt)
        
        print("\n大模型回答:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        
        if response and len(response.strip()) > 0:
            print("✓ 完整流程测试通过！")
            return True
        else:
            print("✗ 大模型返回空响应")
            return False
        
    except Exception as e:
        print(f"✗ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RAG医疗问答系统测试")
    print("=" * 60)
    
    # 运行各项测试
    tests = [
        ("大模型连接", test_llm_connection),
        ("嵌入模型", test_embedding),
        ("向量检索", test_vector_search),
        ("完整流程", test_full_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 打印测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    # 总体结果
    all_passed = all(result for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！系统运行正常。")
    else:
        print("部分测试失败，请检查配置和依赖。")
    print("=" * 60)


if __name__ == "__main__":
    main()
