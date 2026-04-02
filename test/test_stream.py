#!/usr/bin/env python3
"""
测试 Doubao API 流式响应格式
"""
import sys
import os
import requests

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LLM_CONFIG


def test_streaming_response():
    """测试流式响应格式"""
    print("=" * 60)
    print("测试 Doubao API 流式响应格式...")
    print("=" * 60)
    
    # 获取配置
    config = LLM_CONFIG.get('doubao', {})
    api_key = config.get('api_key')
    api_base = config.get('api_base')
    model = config.get('model')
    
    if not api_key or not api_base or not model:
        print("✗ 缺少 Doubao API 配置")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "你好，请简单介绍一下自己。"
                    }
                ]
            }
        ],
        "stream": True
    }
    
    print(f"发送请求到: {api_base}")
    print(f"使用模型: {model}")
    print()
    
    try:
        response = requests.post(api_base, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        print("收到流式响应:")
        print("-" * 60)
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                print(f"原始响应行: {line}")
                
                if line.startswith('data:'):
                    try:
                        import json
                        data = json.loads(line[5:])
                        print(f"解析后的数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
                    except Exception as e:
                        print(f"解析错误: {e}")
                print()
        
    except Exception as e:
        print(f"请求失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_streaming_response()
