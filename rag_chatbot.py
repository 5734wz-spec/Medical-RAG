"""
RAG医疗问答系统主程序
整合所有模块，提供完整的问答功能
"""
import os
import sys
from typing import List, Dict, Any, Generator

# 导入配置
from config import (
    EMBEDDING_CONFIG, 
    VECTOR_DB_CONFIG, 
    LLM_CONFIG, 
    RETRIEVER_CONFIG
)

# 导入各模块
from embedding_model import EmbeddingFactory
from vector_store import FAISSVectorStore
from retriever import MedicalRetriever
from reranker import MedicalReranker
from prompt_builder import MedicalPromptBuilder
from llm_client import LLMFactory
from question_classifier import QuestionClassifier


class RAGChatBot:
    """RAG医疗问答机器人"""
    
    def __init__(self):
        """初始化RAG问答系统"""
        print("=" * 50)
        print("正在初始化RAG医疗问答系统...")
        print("=" * 50)
        
        # 1. 初始化问句分类器（复用原有模块）
        print("\n[1/7] 初始化问句分类器...")
        self.classifier = QuestionClassifier()
        
        # 2. 初始化嵌入模型
        print("\n[2/7] 初始化嵌入模型...")
        self.embedding_model = EmbeddingFactory.create_embedding(EMBEDDING_CONFIG)
        
        # 3. 初始化向量数据库
        print("\n[3/7] 初始化向量数据库...")
        self.vector_store = FAISSVectorStore(**VECTOR_DB_CONFIG)
        
        # 4. 初始化检索器
        print("\n[4/7] 初始化检索器...")
        self.retriever = MedicalRetriever(self.embedding_model, self.vector_store)
        
        # 5. 初始化重排序器
        print("\n[5/7] 初始化重排序器...")
        self.reranker = MedicalReranker(use_cross_encoder=False)
        
        # 6. 初始化Prompt构建器
        print("\n[6/7] 初始化Prompt构建器...")
        self.prompt_builder = MedicalPromptBuilder()
        
        # 7. 初始化大模型客户端
        print("\n[7/7] 初始化大模型客户端...")
        try:
            self.llm_client = LLMFactory.create_client(LLM_CONFIG)
            print("大模型客户端初始化成功！")
        except ValueError as e:
            print(f"警告: {e}")
            print("大模型客户端初始化失败，请检查API密钥配置")
            self.llm_client = None
        
        print("\n" + "=" * 50)
        print("RAG医疗问答系统初始化完成！")
        print("=" * 50)
    
    def chat(self, query: str, use_rag: bool = True, stream: bool = False) -> str:
        """
        单次问答
        
        Args:
            query: 用户问题
            use_rag: 是否使用RAG
            stream: 是否流式输出
        
        Returns:
            回答内容
        """
        if stream:
            return self._chat_stream(query, use_rag)
        else:
            return self._chat_single(query, use_rag)
    
    def _chat_single(self, query: str, use_rag: bool = True) -> str:
        """非流式问答"""
        # 1. 问句分类
        classify_result = self.classifier.classify(query)
        intent = classify_result.get('question_types', ['others'])[0] if classify_result else 'others'
        entities = classify_result.get('args', {}) if classify_result else {}
        
        print(f"\n意图识别: {intent}")
        print(f"实体识别: {entities}")
        
        if not use_rag or not self.llm_client:
            # 不使用RAG，直接调用大模型
            prompt = query
        else:
            # 2. 检索相关文档
            print("\n正在检索相关知识...")
            documents = self.retriever.retrieve_by_intent(query, intent, entities)
            print(f"检索到 {len(documents)} 个相关文档")
            
            if not documents:
                return "抱歉，未检索到相关知识，无法回答您的问题。"
            
            # 3. 重排序
            print("正在重排序...")
            documents = self.reranker.rerank(query, documents, intent)
            
            # 4. 构建Prompt
            print("正在构建Prompt...")
            prompt = self.prompt_builder.build_prompt_by_intent(
                query, documents, intent, entities
            )
        
        # 5. 调用大模型生成回答
        print("正在生成回答...")
        if self.llm_client:
            answer = self.llm_client.chat(prompt)
        else:
            answer = "大模型客户端未初始化，无法生成回答。"
        
        return answer
    
    def _chat_stream(self, query: str, use_rag: bool = True) -> Generator[str, None, None]:
        """流式问答"""
        # 1. 问句分类
        classify_result = self.classifier.classify(query)
        intent = classify_result.get('question_types', ['others'])[0] if classify_result else 'others'
        entities = classify_result.get('args', {}) if classify_result else {}
        
        if not use_rag or not self.llm_client:
            # 不使用RAG，直接调用大模型
            prompt = query
        else:
            # 2. 检索相关文档
            documents = self.retriever.retrieve_by_intent(query, intent, entities)
            
            # 3. 重排序
            if documents:
                documents = self.reranker.rerank(query, documents, intent)
                
                # 4. 构建Prompt
                prompt = self.prompt_builder.build_prompt_by_intent(
                    query, documents, intent, entities
                )
            else:
                # 如果没有检索到文档，仍然调用大模型（基于大模型自身知识）
                prompt = f"请回答以下医疗问题：{query}"
                print(f"[调试] 未检索到相关文档，使用大模型自身知识回答")
        
        # 5. 调用大模型生成回答（流式）
        if self.llm_client:
            try:
                response_generated = False
                for chunk in self.llm_client.chat_stream(prompt):
                    if chunk:
                        yield chunk
                        response_generated = True
                
                if not response_generated:
                    yield "抱歉，大模型未能生成有效回答。"
                    
            except Exception as e:
                print(f"\n[错误] 大模型调用失败: {e}")
                yield f"抱歉，生成回答时出现错误: {str(e)}"
        else:
            yield "大模型客户端未初始化，无法生成回答。"
    
    def chat_with_retrieval(self, query: str) -> Dict[str, Any]:
        """
        问答并返回检索结果
        
        Args:
            query: 用户问题
        
        Returns:
            包含回答和检索结果的字典
        """
        # 1. 问句分类
        classify_result = self.classifier.classify(query)
        intent = classify_result.get('question_types', ['others'])[0] if classify_result else 'others'
        entities = classify_result.get('args', {}) if classify_result else {}
        
        # 2. 检索
        documents = self.retriever.retrieve_by_intent(query, intent, entities)
        
        # 3. 重排序
        if documents:
            documents = self.reranker.rerank(query, documents, intent)
        
        # 4. 构建Prompt
        prompt = self.prompt_builder.build_prompt_by_intent(
            query, documents, intent, entities
        )
        
        # 5. 生成回答
        if self.llm_client:
            answer = self.llm_client.chat(prompt)
        else:
            answer = "大模型客户端未初始化，无法生成回答。"
        
        return {
            'query': query,
            'intent': intent,
            'entities': entities,
            'retrieved_documents': documents,
            'prompt': prompt,
            'answer': answer
        }
    
    def get_vector_db_stats(self) -> Dict[str, Any]:
        """获取向量数据库统计信息"""
        return self.vector_store.get_collection_stats()


def interactive_chat():
    """交互式对话"""
    print("\n" + "=" * 50)
    print("欢迎使用RAG医疗问答系统！")
    print("输入 'quit' 或 'exit' 退出对话")
    print("输入 'stats' 查看向量数据库统计")
    print("=" * 50 + "\n")
    
    # 初始化机器人
    bot = RAGChatBot()
    
    while True:
        try:
            # 获取用户输入
            query = input("\n用户: ").strip()
            
            # 检查退出命令
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n感谢使用，再见！")
                break
            
            # 检查统计命令
            if query.lower() == 'stats':
                stats = bot.get_vector_db_stats()
                print(f"\n向量数据库统计: {stats}")
                continue
            
            # 检查空输入
            if not query:
                continue
            
            # 生成回答
            print("\n助手: ", end="", flush=True)
            
            # 使用流式输出
            try:
                response_text = ""
                chunk_count = 0
                for chunk in bot.chat(query, stream=True):
                    if chunk:
                        print(chunk, end="", flush=True)
                        response_text += chunk
                        chunk_count += 1
                print()  # 换行
                
                if chunk_count == 0:
                    print("\n[调试] 未收到任何响应块")
                else:
                    print(f"\n[调试] 收到 {chunk_count} 个响应块，总长度: {len(response_text)}")
                    
            except Exception as e:
                print(f"\n[错误] 生成回答时出错: {e}")
                import traceback
                traceback.print_exc()
                
        except KeyboardInterrupt:
            print("\n\n感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG医疗问答系统')
    parser.add_argument('--query', type=str, help='单次查询的问题')
    parser.add_argument('--no-rag', action='store_true', help='不使用RAG')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互模式')
    
    args = parser.parse_args()
    
    if args.interactive or not args.query:
        # 交互模式
        interactive_chat()
    else:
        # 单次查询模式
        bot = RAGChatBot()
        answer = bot.chat(args.query, use_rag=not args.no_rag)
        print(f"\n用户: {args.query}")
        print(f"\n助手: {answer}")


if __name__ == '__main__':
    main()
