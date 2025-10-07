import requests
import json
from openai import OpenAI
import time


class SmartAgricultureLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key="7b3117b8126a3cf63eb03eaec74e9d3e5969e464",
            base_url="https://api-i5i2r1vfu2k3f1w7.aistudio-app.com/v1",
            max_retries=3  # 添加重试机制
        )

        # 农业知识库
        self.knowledge_base = {
            "stamen_vs_pistil": """
            雄蕊和雌蕊的区别：
            1. **雄蕊（雄花）**：
               - 位置：通常位于植株顶部
               - 形态：穗状，较长，会产生花粉
               - 功能：产生花粉，传播授粉
               - 处理：在玉米去雄中需要去除

            2. **雌蕊（雌穗）**：
               - 位置：通常位于叶腋处
               - 形态：较短，有丝状花柱
               - 功能：接受花粉，发育成果实
               - 处理：需要保护，避免损伤

            识别技巧：雄蕊在未开放时较小紧实，开放后松散有花粉；雌蕊有丝状结构。
            """,

            "removal_technique": """
            雄蕊去除技术：
            1. **最佳时机**：雄蕊长度1-2厘米，未开放时
            2. **操作方法**：
               - 用拇指和食指轻轻捏住雄蕊基部
               - 向上斜向拔出，避免直拉
               - 检查是否完全去除
            3. **注意事项**：
               - 避免损伤旁边的雌穗
               - 不要使用工具，以免伤及植株
               - 去除后洗手，防止花粉过敏
            """,

            "disease_identification": """
            常见雄蕊异常识别：
            1. **病害表现**：
               - 变色：褐色、黑色斑点
               - 变形：扭曲、萎缩
               - 霉层：白色或黑色霉菌
            2. **虫害表现**：
               - 虫孔：明显的小孔
               - 缺损：部分被啃食
               - 虫粪：黑色颗粒状
            3. **处理建议**：立即去除异常雄蕊，观察邻近植株
            """
        }

    def get_expert_answer(self, question, context=""):
        """获取专家级回答 - 使用新的AI配置"""
        # 构建专业的提示词
        prompt = self._build_prompt(question, context)

        # 添加重试机制
        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model="default",
                    temperature=0.6,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    timeout=30  # 添加超时设置
                )

                # 处理流式响应
                full_response = ""
                for chunk in completion:
                    if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[
                        0].delta.reasoning_content:
                        full_response += chunk.choices[0].delta.reasoning_content
                    else:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content

                # 后处理：移除思考痕迹
                return self._remove_thinking_process(full_response)

            except Exception as e:
                print(f"LLM调用失败 (尝试 {attempt + 1}/3): {e}")
                if attempt < 2:  # 如果不是最后一次尝试，等待后重试
                    time.sleep(2)
                    continue
                else:
                    return self._get_fallback_answer(question, context)

    def _build_prompt(self, question, context):
        """构建专业提示词 - 彻底去除思考过程"""
        base_knowledge = self.knowledge_base.get("stamen_vs_pistil", "")

        prompt = f"""你是一名专业的农业专家。请直接回答问题，不要有任何思考过程、推理步骤或解释。

重要指令：
- 直接给出详细、专业的答案
- 不要使用"我认为"、"首先"、"其次"、"然后"、"最后"等词语
- 不要解释你的思考过程
- 不要使用"思考："、"推理："等前缀
- 答案要详细、直接、专业，可以分点但不使用序号
- 基于检测数据提供具体信息
- 使用清晰的专业语言，避免无关解释

农业专业知识：
{base_knowledge}

当前检测上下文：{context}

用户问题：{question}

直接回答："""

        return prompt

    def _remove_thinking_process(self, answer):
        """移除回答中的思考过程"""
        # 常见的思考模式关键词
        thinking_patterns = [
            "首先，", "其次，", "然后，", "最后，",
            "我认为", "我觉得", "我的看法是",
            "思考：", "推理：", "分析：",
            "让我们来", "我们可以", "应该",
            "总的来说", "综上所述"
        ]

        # 按行分割，移除包含思考模式的行
        lines = answer.split('\n')
        clean_lines = []

        for line in lines:
            line_clean = line.strip()
            # 跳过空行和明显的思考行
            if not line_clean:
                continue

            # 检查是否包含思考模式
            is_thinking = any(pattern in line_clean for pattern in thinking_patterns)

            # 跳过以数字或符号开头的思考列表
            if (line_clean.startswith('1.') or line_clean.startswith('2.') or
                    line_clean.startswith('3.') or line_clean.startswith('- ') or
                    line_clean.startswith('• ')):
                # 但保留真正的要点列表
                if len(line_clean) > 10:  # 确保不是太短的思考点
                    clean_lines.append(line_clean)
            elif not is_thinking and len(line_clean) > 5:  # 确保不是太短的思考片段
                clean_lines.append(line_clean)

        if not clean_lines:
            # 如果全部被过滤掉了，返回原始答案但移除明显的思考词
            for pattern in thinking_patterns:
                answer = answer.replace(pattern, "")
            return answer.strip()

        return '\n'.join(clean_lines)

    def _get_fallback_answer(self, question, context):
        """备用的智能回答（当LLM不可用时）"""
        question_lower = question.lower()

        # 雄蕊雌蕊识别问题
        if any(word in question_lower for word in ['雄蕊', '雌蕊', '区别', '识别', '判断']):
            return """雄蕊识别：
位置：植株顶部
形态：穗状，较长
特征：产生黄色花粉
处理：需要去除

雌蕊识别：
位置：叶腋处
形态：较短，有丝状花柱
特征：发育成玉米棒
处理：必须保护

识别技巧：
观察位置：顶部为雄蕊
触摸感受：雄蕊较硬
发育阶段：雄蕊先出现
天气影响：晴天更易识别
操作时机：雄蕊1-2厘米时最佳"""

        # 去除技术问题
        elif any(word in question_lower for word in ['去除', '方法', '技术', '操作']):
            return """雄蕊去除操作：
最佳时机：雄蕊长度1-2厘米，未开放状态

操作步骤：
定位植株顶部的雄蕊
拇指和食指捏住雄蕊基部
向上斜向轻轻拔出
检查完全去除
处理残留花粉

注意事项：
避免损伤雌穗
不要使用工具
去除后检查遗漏
异常雄蕊单独处理
操作后清理田间"""

        # 病害识别问题
        elif any(word in question_lower for word in ['病害', '虫害', '异常', '病']):
            return """雄蕊异常识别：
病害表现：
变色：褐色、黑色斑点
变形：扭曲、萎缩
霉层：白色或黑色霉菌

虫害表现：
虫孔：明显小孔
缺损：部分被啃食
虫粪：黑色颗粒

处理措施：
立即去除异常雄蕊
隔离处理
观察邻近植株
记录报告异常
加强田间管理"""

        else:
            return f"""问题：{question}

检测情况：{context if context else '暂无检测数据'}

建议：
确保雄蕊识别准确
不确定时咨询技术人员
严格按照操作规程执行
记录操作过程
定期检查效果"""


# 创建全局实例
agriculture_llm = SmartAgricultureLLM()
