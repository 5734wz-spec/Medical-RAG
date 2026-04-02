"""
Prompt构建模块
根据检索结果构建Prompt
"""
from typing import List, Dict, Any
from config import PROMPT_TEMPLATES


class PromptBuilder:
    """Prompt构建器"""
    
    def __init__(self, template_name: str = "medical_qa"):
        """
        初始化
        
        Args:
            template_name: 模板名称
        """
        self.template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES['medical_qa'])
    
    def build_prompt(self, query: str, documents: List[Dict[str, Any]], 
                     intent: str = None) -> str:
        """
        构建Prompt
        
        Args:
            query: 用户查询
            documents: 检索到的文档
            intent: 查询意图
        
        Returns:
            构建好的Prompt
        """
        # 构建上下文
        context = self._build_context(documents, intent)
        
        # 填充模板
        prompt = self.template.format(
            context=context,
            question=query
        )
        
        return prompt
    
    def _build_context(self, documents: List[Dict[str, Any]], 
                      intent: str = None) -> str:
        """
        构建上下文
        
        Args:
            documents: 文档列表
            intent: 查询意图
        
        Returns:
            上下文字符串
        """
        if not documents:
            return "未检索到相关知识。"
        
        context_parts = []
        
        for i, doc in enumerate(documents):
            # 获取文档内容
            content = doc.get('document', '')
            metadata = doc.get('metadata', {})
            
            # 获取文档来源信息
            disease = metadata.get('disease', '')
            field_name = metadata.get('field_name', '')
            
            # 构建文档引用
            if disease and field_name:
                source = f"【{disease} - {field_name}】"
            elif disease:
                source = f"【{disease}】"
            else:
                source = f"【知识 {i+1}】"
            
            # 添加文档到上下文
            context_parts.append(f"{source}\n{content}")
        
        # 合并所有文档
        context = "\n\n".join(context_parts)
        
        return context
    
    def build_prompt_with_history(self, query: str, documents: List[Dict[str, Any]],
                                   history: List[Dict[str, str]], 
                                   intent: str = None) -> str:
        """
        构建带历史对话的Prompt
        
        Args:
            query: 用户查询
            documents: 检索到的文档
            history: 历史对话
            intent: 查询意图
        
        Returns:
            构建好的Prompt
        """
        # 构建上下文
        context = self._build_context(documents, intent)
        
        # 构建历史对话
        history_text = self._build_history(history)
        
        # 构建完整Prompt
        prompt = f"""你是一个专业的医疗知识助手。请基于以下检索到的医疗知识和对话历史，回答用户的问题。

【检索到的相关知识】
{context}

【对话历史】
{history_text}

【用户问题】
{query}

【回答要求】
1. 请基于上述检索到的知识进行回答
2. 结合对话历史，保持回答的连贯性
3. 如果检索知识不足以回答问题，请明确说明
4. 回答要准确、专业、易懂
5. 如有多个相关信息，请分点说明
6. 不要编造不在检索知识中的内容

请回答："""
        
        return prompt
    
    def _build_history(self, history: List[Dict[str, str]]) -> str:
        """
        构建历史对话文本
        
        Args:
            history: 历史对话列表
        
        Returns:
            历史对话文本
        """
        if not history:
            return "无历史对话"
        
        history_parts = []
        for turn in history:
            user_msg = turn.get('user', '')
            assistant_msg = turn.get('assistant', '')
            history_parts.append(f"用户：{user_msg}")
            history_parts.append(f"助手：{assistant_msg}")
        
        return "\n".join(history_parts)
    
    def set_template(self, template_name: str):
        """
        设置模板
        
        Args:
            template_name: 模板名称
        """
        if template_name in PROMPT_TEMPLATES:
            self.template = PROMPT_TEMPLATES[template_name]
        else:
            raise ValueError(f"未知的模板名称: {template_name}")


class MedicalPromptBuilder(PromptBuilder):
    """医疗专用Prompt构建器"""
    
    def __init__(self):
        super().__init__("medical_qa")
    
    def build_prompt_by_intent(self, query: str, documents: List[Dict[str, Any]],
                                intent: str, entities: Dict[str, List[str]] = None) -> str:
        """
        根据意图构建Prompt
        
        Args:
            query: 用户查询
            documents: 检索到的文档
            intent: 查询意图
            entities: 识别的实体
        
        Returns:
            构建好的Prompt
        """
        # 根据意图选择模板
        template_map = {
            'disease_symptom': 'disease_symptom',
            'disease_cause': 'disease_symptom',
            'disease_prevent': 'disease_symptom',
            'disease_cureway': 'disease_treatment',
            'disease_cureprob': 'disease_treatment',
            'disease_lasttime': 'disease_treatment',
            'disease_easyget': 'disease_symptom',
            'disease_acompany': 'disease_symptom',
            'disease_not_food': 'disease_symptom',
            'disease_do_food': 'disease_symptom',
            'disease_drug': 'disease_treatment',
            'disease_check': 'disease_symptom',
            'disease_desc': 'medical_qa',
            'symptom_disease': 'medical_qa',
        }
        
        template_name = template_map.get(intent, 'medical_qa')
        self.set_template(template_name)
        
        # 构建上下文
        context = self._build_context(documents, intent)
        
        # 添加实体信息
        entity_info = ""
        if entities:
            entity_parts = []
            if 'disease' in entities:
                entity_parts.append(f"疾病：{', '.join(entities['disease'])}")
            if 'symptom' in entities:
                entity_parts.append(f"症状：{', '.join(entities['symptom'])}")
            if entity_parts:
                entity_info = "【识别的实体】\n" + "\n".join(entity_parts) + "\n\n"
        
        # 填充模板
        prompt = self.template.format(
            context=entity_info + context,
            question=query
        )
        
        return prompt


if __name__ == '__main__':
    # 测试代码
    builder = MedicalPromptBuilder()
    
    # 测试数据
    query = "高血压的症状有哪些？"
    documents = [
        {
            'document': '高血压的症状包括头痛、头晕、心悸、胸闷等。部分患者可能没有明显症状。',
            'metadata': {'disease': '高血压', 'field_name': '症状'}
        },
        {
            'document': '长期高血压可导致心脑血管疾病，需要及时治疗。',
            'metadata': {'disease': '高血压', 'field_name': '简介'}
        }
    ]
    
    # 测试基本Prompt构建
    print("=== 基本Prompt ===")
    prompt = builder.build_prompt(query, documents)
    print(prompt[:500] + "...")
    
    # 测试按意图构建
    print("\n=== 按意图构建Prompt ===")
    prompt = builder.build_prompt_by_intent(
        query, 
        documents, 
        intent='disease_symptom',
        entities={'disease': ['高血压']}
    )
    print(prompt[:500] + "...")
