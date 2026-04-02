"""
大模型客户端模块
支持多种大模型API：OpenAI、通义千问、文心一言、智谱AI等
"""
import os
import json
import requests
from typing import List, Dict, Any, Generator
from abc import ABC, abstractmethod

from config import LLM_CONFIG


class BaseLLMClient(ABC):
    """大模型客户端基类"""
    
    @abstractmethod
    def chat(self, prompt: str, **kwargs) -> str:
        """单次对话"""
        pass
    
    @abstractmethod
    def chat_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端"""
    
    def __init__(self, api_key: str, api_base: str, model: str = "gpt-3.5-turbo",
                 temperature: float = 0.7, max_tokens: int = 2000, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
        
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
    
    def chat(self, prompt: str, **kwargs) -> str:
        """单次对话"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            **self.extra_params
        )
        return response.choices[0].message.content
    
    def chat_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            stream=True,
            **self.extra_params
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class QwenClient(BaseLLMClient):
    """通义千问客户端"""
    
    def __init__(self, api_key: str, model: str = "qwen-turbo",
                 temperature: float = 0.7, max_tokens: int = 2000, **kwargs):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    def chat(self, prompt: str, **kwargs) -> str:
        """单次对话"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "temperature": kwargs.get('temperature', self.temperature),
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "result_format": "message"
            }
        }
        
        response = requests.post(self.api_base, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        return result['output']['choices'][0]['message']['content']
    
    def chat_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "enable"
        }
        
        data = {
            "model": self.model,
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "temperature": kwargs.get('temperature', self.temperature),
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "incremental_output": True,
                "result_format": "message"
            }
        }
        
        response = requests.post(self.api_base, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data:'):
                    try:
                        data = json.loads(line[5:])
                        if 'output' in data and 'choices' in data['output']:
                            content = data['output']['choices'][0]['message']['content']
                            yield content
                    except:
                        pass


class WenxinClient(BaseLLMClient):
    """文心一言客户端"""
    
    def __init__(self, api_key: str, secret_key: str, model: str = "ernie-bot-turbo",
                 temperature: float = 0.7, max_tokens: int = 2000, **kwargs):
        self.api_key = api_key
        self.secret_key = secret_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.access_token = self._get_access_token()
    
    def _get_access_token(self) -> str:
        """获取访问令牌"""
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.secret_key}"
        response = requests.post(url)
        response.raise_for_status()
        return response.json()['access_token']
    
    def chat(self, prompt: str, **kwargs) -> str:
        """单次对话"""
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.model}?access_token={self.access_token}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', self.temperature),
            "max_output_tokens": kwargs.get('max_tokens', self.max_tokens)
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        return result['result']
    
    def chat_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        # 文心一言流式API实现
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.model}?access_token={self.access_token}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', self.temperature),
            "max_output_tokens": kwargs.get('max_tokens', self.max_tokens),
            "stream": True
        }
        
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'result' in data:
                        yield data['result']
                except:
                    pass


class ZhipuClient(BaseLLMClient):
    """智谱AI客户端"""
    
    def __init__(self, api_key: str, model: str = "chatglm_turbo",
                 temperature: float = 0.7, max_tokens: int = 2000, **kwargs):
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            raise ImportError("请安装zhipuai库: pip install zhipuai")
        
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def chat(self, prompt: str, **kwargs) -> str:
        """单次对话"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens)
        )
        return response.choices[0].message.content
    
    def chat_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaClient(BaseLLMClient):
    """Ollama本地模型客户端（完全免费）"""
    
    def __init__(self, model: str = "llama3:8b", api_base: str = "http://localhost:11434",
                 temperature: float = 0.7, max_tokens: int = 2000, **kwargs):
        self.model = model
        self.api_base = api_base.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def chat(self, prompt: str, **kwargs) -> str:
        """单次对话"""
        import requests
        
        url = f"{self.api_base}/api/chat"
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "stream": False
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        return result['message']['content']
    
    def chat_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        import requests
        
        url = f"{self.api_base}/api/chat"
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "stream": True
        }
        
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                import json
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'message' in data and 'content' in data['message']:
                        content = data['message']['content']
                        if content:
                            yield content
                except:
                    pass


class DoubaoClient(BaseLLMClient):
    """字节跳动Doubao客户端（有免费额度）"""
    
    def __init__(self, api_key: str, model: str = "",
                 temperature: float = 0.7, max_tokens: int = 2000, **kwargs):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = ""
    
    def chat(self, prompt: str, **kwargs) -> str:
        """单次对话"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(self.api_base, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        # 解析响应结果
        try:
            # 新的Doubao API格式
            if 'output' in result and isinstance(result['output'], list):
                # 查找message类型的输出
                for output_item in result['output']:
                    if output_item.get('type') == 'message' and 'content' in output_item:
                        content = output_item['content']
                        if isinstance(content, list) and len(content) > 0:
                            return content[0].get('text', '')
                        elif isinstance(content, str):
                            return content
            
            # 旧的Doubao API格式
            if 'output' in result and isinstance(result['output'], dict):
                if 'text' in result['output']:
                    return result['output']['text']
            
            # 兼容OpenAI格式
            if 'choices' in result and result['choices']:
                return result['choices'][0]['message']['content']
            
            # 如果都解析失败，返回错误信息
            raise ValueError(f"无法解析Doubao API响应: {result}")
            
        except Exception as e:
            raise ValueError(f"解析Doubao API响应失败: {e}, 响应内容: {result}")
    
    def chat_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        try:
            # 直接调用非流式方法获取完整响应
            response = self.chat(prompt, **kwargs)
            if response:
                # 模拟流式输出，一次性返回完整内容
                yield response
            else:
                yield "抱歉，大模型未能生成有效回答。"
        except Exception as e:
            print(f"[错误] 大模型流式调用失败: {e}")
            yield f"抱歉，生成回答时出现错误: {str(e)}"


class GeminiClient(BaseLLMClient):
    """Google Gemini客户端（有免费额度）"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash",
                 temperature: float = 0.7, max_tokens: int = 2000, **kwargs):
        try:
            from google.generativeai import GenerativeModel
        except ImportError:
            raise ImportError("请安装google-generativeai库: pip install google-generativeai")
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        self.model = GenerativeModel(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def chat(self, prompt: str, **kwargs) -> str:
        """单次对话"""
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": kwargs.get('temperature', self.temperature),
                "max_output_tokens": kwargs.get('max_tokens', self.max_tokens)
            }
        )
        return response.text
    
    def chat_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": kwargs.get('temperature', self.temperature),
                "max_output_tokens": kwargs.get('max_tokens', self.max_tokens)
            },
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text


class LLMFactory:
    """大模型客户端工厂"""
    
    @staticmethod
    def create_client(config: dict = None) -> BaseLLMClient:
        """
        根据配置创建大模型客户端
        
        Args:
            config: 配置字典，来自config.py中的LLM_CONFIG
        
        Returns:
            大模型客户端实例
        """
        if config is None:
            config = LLM_CONFIG
        
        model_type = config.get('model_type', 'openai')
        
        if model_type == 'openai':
            openai_config = config.get('openai', {})
            api_key = openai_config.get('api_key')
            if not api_key:
                raise ValueError("OpenAI API密钥未设置，请设置OPENAI_API_KEY环境变量")
            
            return OpenAIClient(
                api_key=api_key,
                api_base=openai_config.get('api_base', 'https://api.openai.com/v1'),
                model=openai_config.get('model', 'gpt-3.5-turbo'),
                temperature=openai_config.get('temperature', 0.7),
                max_tokens=openai_config.get('max_tokens', 2000),
                top_p=openai_config.get('top_p', 0.9)
            )
        
        elif model_type == 'qwen':
            qwen_config = config.get('qwen', {})
            api_key = qwen_config.get('api_key')
            if not api_key:
                raise ValueError("通义千问API密钥未设置，请设置DASHSCOPE_API_KEY环境变量")
            
            return QwenClient(
                api_key=api_key,
                model=qwen_config.get('model', 'qwen-turbo'),
                temperature=qwen_config.get('temperature', 0.7),
                max_tokens=qwen_config.get('max_tokens', 2000)
            )
        
        elif model_type == 'wenxin':
            wenxin_config = config.get('wenxin', {})
            api_key = wenxin_config.get('api_key')
            secret_key = wenxin_config.get('secret_key')
            if not api_key or not secret_key:
                raise ValueError("文心一言API密钥未设置，请设置WENXIN_API_KEY和WENXIN_SECRET_KEY环境变量")
            
            return WenxinClient(
                api_key=api_key,
                secret_key=secret_key,
                model=wenxin_config.get('model', 'ernie-bot-turbo'),
                temperature=wenxin_config.get('temperature', 0.7),
                max_tokens=wenxin_config.get('max_tokens', 2000)
            )
        
        elif model_type == 'zhipu':
            zhipu_config = config.get('zhipu', {})
            api_key = zhipu_config.get('api_key')
            if not api_key:
                raise ValueError("智谱AI API密钥未设置，请设置ZHIPU_API_KEY环境变量")
            
            return ZhipuClient(
                api_key=api_key,
                model=zhipu_config.get('model', 'chatglm_turbo'),
                temperature=zhipu_config.get('temperature', 0.7),
                max_tokens=zhipu_config.get('max_tokens', 2000)
            )
        
        elif model_type == 'ollama':
            ollama_config = config.get('ollama', {})
            return OllamaClient(
                model=ollama_config.get('model', 'llama3:8b'),
                api_base=ollama_config.get('api_base', 'http://localhost:11434'),
                temperature=ollama_config.get('temperature', 0.7),
                max_tokens=ollama_config.get('max_tokens', 2000)
            )
        
        elif model_type == 'doubao':
            doubao_config = config.get('doubao', {})
            api_key = doubao_config.get('api_key')
            if not api_key:
                raise ValueError("Doubao API密钥未设置，请设置DOUBAO_API_KEY环境变量")
            
            return DoubaoClient(
                api_key=api_key,
                model=doubao_config.get('model', 'ep-20260330172723-q47qh'),
                temperature=doubao_config.get('temperature', 0.7),
                max_tokens=doubao_config.get('max_tokens', 2000)
            )
        
        elif model_type == 'gemini':
            gemini_config = config.get('gemini', {})
            api_key = gemini_config.get('api_key')
            if not api_key:
                raise ValueError("Gemini API密钥未设置，请设置GEMINI_API_KEY环境变量")
            
            return GeminiClient(
                api_key=api_key,
                model=gemini_config.get('model', 'gemini-1.5-flash'),
                temperature=gemini_config.get('temperature', 0.7),
                max_tokens=gemini_config.get('max_tokens', 2000)
            )
        
        else:
            raise ValueError(f"不支持的大模型类型: {model_type}")


if __name__ == '__main__':
    # 测试代码
    from config import LLM_CONFIG
    
    # 创建客户端
    try:
        client = LLMFactory.create_client(LLM_CONFIG)
        
        # 测试对话
        prompt = "你好，请简单介绍一下自己。"
        print(f"用户: {prompt}")
        print("助手: ", end="", flush=True)
        
        # 流式输出
        for chunk in client.chat_stream(prompt):
            print(chunk, end="", flush=True)
        print()
        
    except ValueError as e:
        print(f"错误: {e}")
        print("请设置相应的API密钥环境变量后再测试")
