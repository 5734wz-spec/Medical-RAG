"""
工具函数模块
提供通用的工具函数
"""
import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime


def setup_logging(log_file: str = None, level: str = "INFO"):
    """
    设置日志
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
    """
    from config import LOG_CONFIG
    
    if log_file is None:
        log_file = LOG_CONFIG.get('file', 'rag.log')
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format=LOG_CONFIG.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    加载JSONL文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """
    保存为JSONL文件
    
    Args:
        data: 数据列表
        file_path: 文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def format_medical_answer(answer: str, max_length: int = 500) -> str:
    """
    格式化医疗回答
    
    Args:
        answer: 原始回答
        max_length: 最大长度
    
    Returns:
        格式化后的回答
    """
    # 去除多余空白
    answer = ' '.join(answer.split())
    
    # 限制长度
    if len(answer) > max_length:
        answer = answer[:max_length] + "..."
    
    return answer


def highlight_keywords(text: str, keywords: List[str]) -> str:
    """
    高亮关键词
    
    Args:
        text: 原文本
        keywords: 关键词列表
    
    Returns:
        高亮后的文本
    """
    for keyword in keywords:
        text = text.replace(keyword, f"**{keyword}**")
    return text


def calculate_similarity(text1: str, text2: str) -> float:
    """
    计算文本相似度（简单版本）
    
    Args:
        text1: 文本1
        text2: 文本2
    
    Returns:
        相似度分数 (0-1)
    """
    # 使用Jaccard相似度
    set1 = set(text1)
    set2 = set(text2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本
    
    Args:
        text: 原文本
        max_length: 最大长度
        suffix: 后缀
    
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    合并两个字典
    
    Args:
        dict1: 字典1
        dict2: 字典2
    
    Returns:
        合并后的字典
    """
    result = dict1.copy()
    result.update(dict2)
    return result


def get_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(dir_path: str):
    """确保目录存在"""
    os.makedirs(dir_path, exist_ok=True)


def read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """读取文件内容"""
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def write_file(file_path: str, content: str, encoding: str = 'utf-8'):
    """写入文件内容"""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)


class ProgressBar:
    """简单的进度条"""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
    
    def update(self, n: int = 1):
        """更新进度"""
        self.current += n
        percent = (self.current / self.total) * 100
        print(f"\r{self.desc}: [{self.current}/{self.total}] {percent:.1f}%", end="", flush=True)
        if self.current >= self.total:
            print()
    
    def close(self):
        """关闭进度条"""
        if self.current < self.total:
            print()


if __name__ == '__main__':
    # 测试代码
    print("=== 工具函数测试 ===")
    
    # 测试日志
    logger = setup_logging()
    logger.info("日志测试")
    
    # 测试文本截断
    text = "这是一个很长的文本，需要被截断"
    print(f"\n截断文本: {truncate_text(text, max_length=10)}")
    
    # 测试相似度计算
    text1 = "高血压的症状"
    text2 = "高血压有哪些症状"
    print(f"\n相似度: {calculate_similarity(text1, text2):.4f}")
    
    # 测试时间戳
    print(f"\n时间戳: {get_timestamp()}")
    
    print("\n测试完成！")
