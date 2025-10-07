import os
import tempfile
import uuid
import wave
import threading
voice_lock = threading.Lock()
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import math
import gc
import threading
import time
import re
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageEnhance

# 设置环境变量
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

TITLE = "🌽 智能农业助手"
DESCRIPTION = "玉米去雄智能识别与指导系统"

# 导入模块
from smart_agriculture_llm import agriculture_llm

# 检测历史存储
detection_history = []

# 知识图谱历史存储
knowledge_graph_history = []

# 语音处理器（懒加载）
_speech_processor = None

# 性能监控
performance_stats = {
    "last_cleanup": datetime.now(),
    "generated_files": set(),
    "active_threads": 0
}

# 全局字体缓存
_font_cache = None
_font_small_cache = None


# 知识图谱相关函数
def save_knowledge_graph_record(question, answer, kg_image_path):
    """保存知识图谱记录"""
    if not kg_image_path or not os.path.exists(kg_image_path):
        return None

    record = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer[:200] + "..." if len(answer) > 200 else answer,  # 截断长答案
        'kg_image': kg_image_path
    }
    knowledge_graph_history.append(record)

    # 限制历史记录数量，避免内存泄漏
    if len(knowledge_graph_history) > 100:
        oldest_record = knowledge_graph_history.pop(0)
        # 清理旧文件
        try:
            if os.path.exists(oldest_record['kg_image']):
                os.remove(oldest_record['kg_image'])
        except:
            pass

    return record


def get_knowledge_graph_gallery():
    """获取知识图谱画廊数据"""
    if not knowledge_graph_history:
        return [], "暂无知识图谱记录"

    gallery_data = []
    for record in reversed(knowledge_graph_history[-20:]):  # 只显示最近20个
        if os.path.exists(record['kg_image']):
            gallery_data.append((record['kg_image'], f"Q: {record['question'][:30]}..."))

    return gallery_data, f"共 {len(knowledge_graph_history)} 个知识图谱"


def get_knowledge_graph_detail(kg_image_path):
    """获取知识图谱详细信息"""
    if not kg_image_path:
        return "请选择知识图谱", "", ""

    for record in knowledge_graph_history:
        if record['kg_image'] == kg_image_path:
            return record['question'], record['answer'], record['timestamp']

    return "未找到详细信息", "", ""


def cleanup_old_files():
    """清理旧文件，释放资源"""
    try:
        current_time = datetime.now()
        if (current_time - performance_stats["last_cleanup"]).seconds < 300:
            return

        print("🧹 清理旧文件...")
        temp_dir = tempfile.gettempdir()

        # 清理临时图片文件（保留知识图谱文件）
        kg_files = {record['kg_image'] for record in knowledge_graph_history}

        for filename in performance_stats["generated_files"].copy():
            filepath = os.path.join(temp_dir, filename)
            if os.path.exists(filepath) and filepath not in kg_files:
                try:
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    if (current_time - file_time).seconds > 600:
                        os.remove(filepath)
                        performance_stats["generated_files"].remove(filename)
                        print(f"🗑️ 删除旧文件: {filename}")
                except Exception as e:
                    print(f"删除文件失败 {filename}: {e}")

        # 强制垃圾回收
        gc.collect()
        performance_stats["last_cleanup"] = current_time
        print("✅ 清理完成")

    except Exception as e:
        print(f"清理文件时出错: {e}")


def get_speech_processor():
    """获取语音处理器实例（懒加载）"""
    global _speech_processor
    if _speech_processor is None:
        try:
            from speech_processor import speech_processor as sp
            _speech_processor = sp
            print("✅ 语音处理器加载成功")

            # 测试语音合成引擎
            if _speech_processor.tts_engine:
                print("✅ 语音合成引擎可用")
            else:
                print("❌ 语音合成引擎不可用")

        except Exception as e:
            print(f"❌ 语音处理器加载失败: {e}")

            # 创建改进的虚拟语音处理器
            class DummySpeechProcessor:
                def __init__(self):
                    self.tts_engine = None
                    print("⚠️ 使用虚拟语音处理器")

                def speech_to_text(self, audio_file):
                    return "语音识别功能暂不可用，请使用文本输入"

                def text_to_speech(self, text, save_to_file=False):
                    print(f"🔊 虚拟语音合成: {text[:50]}...")
                    if save_to_file:
                        # 创建一个空的音频文件作为占位符
                        temp_file = os.path.join(tempfile.gettempdir(), f"tts_dummy_{uuid.uuid4().hex}.wav")
                        # 创建一个空的WAV文件
                        with wave.open(temp_file, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(16000)
                            wf.writeframes(b'')
                        print(f"📁 创建虚拟音频文件: {temp_file}")
                        return temp_file
                    return None

            _speech_processor = DummySpeechProcessor()
    return _speech_processor


def save_detection_record(detection_data, area_info=""):
    """保存检测记录"""
    record = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'area': area_info,
        'data': detection_data
    }
    detection_history.append(record)

    # 限制历史记录数量，避免内存泄漏
    if len(detection_history) > 50:
        detection_history.pop(0)

    return record


def preprocess_image(img):
    """预处理图像，确保格式正确"""
    if img is None:
        return None

    try:
        # 如果已经是PIL图像，直接返回
        if isinstance(img, Image.Image):
            return img

        # 如果是numpy数组，转换为PIL图像
        import numpy as np
        if isinstance(img, np.ndarray):
            # 确保是uint8类型
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return Image.fromarray(img)

        # 如果是文件路径，打开图像
        if isinstance(img, str) and os.path.exists(img):
            return Image.open(img)

        # 如果是Gradio文件对象
        if hasattr(img, 'name') and os.path.exists(img.name):
            return Image.open(img.name)

    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None

    return None


def safe_image_save(image, path, quality=85):
    """安全保存图像，避免权限问题"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path, quality=quality, optimize=True)

        # 记录生成的文件
        filename = os.path.basename(path)
        performance_stats["generated_files"].add(filename)

        return path
    except Exception as e:
        print(f"保存图像失败: {e}")
        # 使用临时文件作为备选
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.jpg")
        image.save(temp_path, quality=quality, optimize=True)

        filename = os.path.basename(temp_path)
        performance_stats["generated_files"].add(filename)

        return temp_path


def fast_inference(img, threshold, area_info, enable_guidance):
    """快速检测推理函数 - 优化处理速度"""
    processed_img = preprocess_image(img)
    if processed_img is None:
        return None, "请先上传有效的图像文件", ""

    try:
        # 记录开始时间
        start_time = datetime.now()

        # 执行检测
        from Stamen_detection import detect_stamen
        boxed_pil, detection_result = detect_stamen(processed_img, threshold)

        # 计算处理时间
        process_time = (datetime.now() - start_time).total_seconds()

        # 生成检测结果文本
        detection_text = detection_result.to_text_summary()

        # 添加位置信息
        if hasattr(detection_result, 'to_position_summary'):
            position_text = detection_result.to_position_summary()
            full_detection_text = f"{detection_text}\n\n{position_text}"
        else:
            full_detection_text = detection_text

        # 添加处理时间信息
        full_detection_text = f"⏱️ 处理时间: {process_time:.2f}秒\n\n{full_detection_text}"

        # 保存检测记录
        detection_data = detection_result.get_detection_data()
        save_detection_record(detection_data, area_info)

        # 生成操作指导 - 使用线程避免阻塞
        guidance_text = ""

        if enable_guidance and detection_result.detected_objects:
            try:
                # 使用线程生成指导，避免阻塞主线程
                def generate_guidance_thread():
                    nonlocal guidance_text
                    try:
                        guidance_prompt = f"""
检测结果：{detection_text}

请直接给出具体的去雄操作指导，要求：
- 直接给出操作步骤，不要有任何思考过程
- 步骤清晰实用，易于执行
- 基于检测结果针对性建议
- 使用通俗易懂的语言
- 不超过5个步骤

操作指导：
"""
                        guidance_response = agriculture_llm.get_expert_answer(guidance_prompt, detection_text)
                        guidance_text = clean_ai_response(guidance_response)
                    except Exception as e:
                        print(f"生成指导失败: {e}")
                        guidance_text = "AI指导暂时不可用，请参考标准操作流程。"

                # 启动线程，但设置超时
                guidance_thread = threading.Thread(target=generate_guidance_thread)
                guidance_thread.daemon = True
                guidance_thread.start()
                guidance_thread.join(timeout=10)  # 10秒超时

                if guidance_thread.is_alive():
                    guidance_text = "操作指导生成超时，请稍后重试"

            except Exception as e:
                print(f"生成指导失败: {e}")
                guidance_text = "AI指导暂时不可用，请参考标准操作流程。"

        # 保存结果图像
        out_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.jpg")
        out_path = safe_image_save(boxed_pil, out_path, quality=85)

        print(f"快速检测完成，发现 {len(detection_result.detected_objects)} 个雄蕊，耗时 {process_time:.2f}秒")
        return out_path, full_detection_text, guidance_text

    except Exception as e:
        error_msg = f"检测失败: {str(e)}"
        print(f"错误详情: {e}")

        # 提供友好的错误信息
        if "timeout" in str(e).lower() or "超时" in str(e):
            error_msg += "\n\n建议：网络连接较慢，请稍后重试"
        elif "network" in str(e).lower() or "连接" in str(e):
            error_msg += "\n\n建议：请检查网络连接后重试"
        else:
            error_msg += "\n\n建议：请稍后重试或联系技术支持"

        return None, error_msg, ""


def clean_ai_response(response):
    """彻底清理AI响应，去除所有思考过程和内部推理"""
    if not response:
        return response

    # 定义需要过滤的思考模式关键词
    thought_patterns = [
        '用户问的是', '根据提供的', '检测数据里提到', '所以直接回答',
        '不需要额外解释', '按照要求', '这部分明显', '可能还要考虑',
        '需要确保', '可能存在的问题', '改进建议包括', '下一步工作计划',
        '用户要求', '需要确保每个部分', '同时要避免', '直接列出分析结果',
        '思考：', '分析：', '推理：', '首先', '然后', '接着', '最后',
        '基于以上', '综上所述', '总而言之', '因此', '所以', '因而',
        '从数据可以看出', '根据分析结果', '我认为', '我建议', '我的看法是',
        '让我们来', '接下来我们', '现在开始', '开始回答', '回答如下',
        '操作指导部分', '步骤要简短实用', '这部分才是回答', '综合检测报告部分',
        '雄蕊分布特点分析', '可能存在的问题方面', '改进建议包括', '下一步工作计划需要'
    ]

    lines = response.split('\n')
    cleaned_lines = []
    in_thought_process = False

    for line in lines:
        line = line.strip()

        # 跳过空行
        if not line:
            continue

        # 检测是否进入思考过程段落
        if any(pattern in line for pattern in thought_patterns):
            in_thought_process = True
            continue

        # 如果检测到实际内容开始，重置状态
        if line and not in_thought_process:
            # 跳过数字序号但保留内容（如"1. "变成空）
            if re.match(r'^\d+\.\s', line):
                line = re.sub(r'^\d+\.\s', '', line)
            cleaned_lines.append(line)
        elif in_thought_process and line and not any(pattern in line for pattern in thought_patterns):
            # 如果遇到新段落，退出思考过程模式
            in_thought_process = False
            cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines)

    # 如果清理后为空，返回原始响应但进行基础清理
    if not result.strip():
        # 基础清理：移除明显的思考标记
        result = re.sub(r'思考：.*?\n', '', response)
        result = re.sub(r'分析：.*?\n', '', result)
        result = re.sub(r'推理：.*?\n', '', result)

    return result.strip()


def draw_arrow_simple(draw, x1, y1, x2, y2, color=(100, 100, 100)):
    """简化的箭头绘制"""
    # 计算方向
    dx, dy = x2 - x1, y2 - y1
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        return

    # 归一化
    dx, dy = dx / length, dy / length

    # 箭头位置（稍微向内）
    arrow_x = x2 - dx * 15
    arrow_y = y2 - dy * 15

    # 绘制箭头三角形
    size = 8
    perpendicular_x = -dy * size
    perpendicular_y = dx * size

    points = [
        (arrow_x, arrow_y),
        (arrow_x - dx * size + perpendicular_x, arrow_y - dy * size + perpendicular_y),
        (arrow_x - dx * size - perpendicular_x, arrow_y - dy * size - perpendicular_y)
    ]

    draw.polygon(points, fill=color)


def get_fonts():
    """获取字体 - 使用缓存避免重复加载，优先中文字体"""
    global _font_cache, _font_small_cache

    if _font_cache is not None and _font_small_cache is not None:
        return _font_cache, _font_small_cache

    try:
        from PIL import ImageFont

        # 优先尝试中文字体
        chinese_font_paths = [
            "C:/Windows/Fonts/simsun.ttc",  # Windows 宋体
            "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
            "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
        ]

        font_loaded = False
        for font_path in chinese_font_paths:
            if os.path.exists(font_path):
                try:
                    _font_cache = ImageFont.truetype(font_path, 16)
                    _font_small_cache = ImageFont.truetype(font_path, 12)
                    print(f"✅ 中文字体加载成功: {font_path}")
                    font_loaded = True
                    break
                except Exception as e:
                    print(f"字体加载失败 {font_path}: {e}")
                    continue

        # 如果中文字体都失败，使用默认字体
        if not font_loaded:
            try:
                _font_cache = ImageFont.load_default()
                _font_small_cache = ImageFont.load_default()
                print("⚠️ 使用默认字体（可能不支持中文）")
            except:
                _font_cache = None
                _font_small_cache = None

    except Exception as e:
        print(f"字体系统加载失败: {e}")
        _font_cache = None
        _font_small_cache = None

    return _font_cache, _font_small_cache


def get_detection_context():
    """获取当前检测上下文数据"""
    if not detection_history:
        return "暂无检测数据，请先进行图像检测"

    try:
        # 获取最新的检测结果
        latest_record = detection_history[-1]
        detection_data = latest_record['data']

        # 构建检测上下文
        context_parts = []

        # 基础统计信息
        if 'total_count' in detection_data:
            context_parts.append(f"检测到雄蕊数量: {detection_data['total_count']}个")

        if 'detected_objects' in detection_data:
            objects = detection_data['detected_objects']
            if objects:
                # 位置信息
                positions = []
                for obj in objects[:5]:  # 只取前5个避免太长
                    if 'position' in obj:
                        pos = obj['position']
                        positions.append(f"({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f})")

                if positions:
                    context_parts.append(f"雄蕊位置: {', '.join(positions)}")

        # 置信度信息
        if 'confidence_scores' in detection_data:
            confidences = detection_data['confidence_scores']
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                context_parts.append(f"平均置信度: {avg_confidence:.2f}")

        # 区域信息
        area = latest_record.get('area', '')
        if area:
            context_parts.append(f"检测区域: {area}")

        return " | ".join(context_parts) if context_parts else "检测数据可用但信息较少"

    except Exception as e:
        print(f"获取检测上下文失败: {e}")
        return "检测数据解析失败"

# 添加缺失的函数
def update_detection_status():
    """更新检测状态显示"""
    context = get_detection_context()
    return f"📊 当前检测状态: {context}"

def handle_voice_question_with_context(audio_file, enable_voice_answer, generate_kg):
    """处理语音提问 - 结合检测上下文，修复语音合成"""
    if audio_file is None:
        return "请先录制语音问题", None, None, "", ""

    try:
        # 语音转文本
        speech_processor = get_speech_processor()
        question_text = speech_processor.speech_to_text(audio_file)
        print(f"语音识别结果: {question_text}")

        if not question_text or "未识别" in question_text or "暂不可用" in question_text:
            return "语音识别失败，请重新录制或使用文本输入", None, None, "", question_text

        # 获取检测上下文
        detection_context = get_detection_context()
        print(f"检测上下文: {detection_context}")

        # 构建智能提示词，结合检测数据
        enhanced_prompt = f"""
用户提问：{question_text}

当前检测数据：{detection_context}

请结合检测数据直接回答用户问题。要求：
- 直接给出答案，不要有任何思考过程或解释
- 简洁明了，回答核心问题
- 如果使用检测数据，请明确指出
- 提供实用的操作建议
- 可以适当扩展专业知识
- 使用通俗易懂的语言

直接给出回答：
"""

        # 使用增强的提示词获取回答
        answer_response = agriculture_llm.get_expert_answer(enhanced_prompt, question_text)
        # 清理响应，去除思考过程
        answer = clean_ai_response(answer_response)

        print(f"✅ AI回答生成完成，长度: {len(answer)}")

        # 语音回答
        audio_file_output = None
        if enable_voice_answer and answer:
            print("🔊 开始语音合成...")
            try:
                speech_processor = get_speech_processor()
                if speech_processor and hasattr(speech_processor, 'tts_engine') and speech_processor.tts_engine:
                    audio_result = [None]

                    # 在 handle_question_with_context 和 handle_voice_question_with_context 中
                    # 修改语音合成部分：

                    def generate_audio():
                        try:
                            speech_processor = get_speech_processor()
                            # 使用修复后的 text_to_speech 方法
                            audio_file_path = speech_processor.text_to_speech(answer, save_to_file=True)
                            if audio_file_path and os.path.exists(audio_file_path):
                                filename = os.path.basename(audio_file_path)
                                performance_stats["generated_files"].add(filename)
                                audio_result[0] = audio_file_path
                                print(f"✅ 语音文件生成成功: {audio_file_path}")
                            else:
                                print("❌ 语音文件生成失败")
                        except Exception as e:
                            print(f"❌ 语音生成线程异常: {e}")

                    audio_thread = threading.Thread(target=generate_audio)
                    audio_thread.daemon = True
                    audio_thread.start()
                    audio_thread.join(timeout=15)

                    audio_file_output = audio_result[0]

                    if not audio_file_output:
                        print("⚠️ 语音生成超时或失败")
                else:
                    print("❌ 语音合成引擎不可用")
            except Exception as e:
                print(f"❌ 语音合成处理失败: {e}")

        # 知识图谱生成 - 修改为保存到历史记录
        kg_status = ""
        if generate_kg and answer:
            try:
                kg_image, kg_status = generate_knowledge_graph_optimized(question_text, answer)
                if kg_image:
                    # 保存到知识图谱历史记录
                    save_knowledge_graph_record(question_text, answer, kg_image)
                    kg_status = "✅ 知识图谱已生成并保存到知识图谱库"
                else:
                    kg_status = "❌ 知识图谱生成失败"
            except Exception as e:
                print(f"知识图谱生成失败: {e}")
                kg_status = "知识图谱生成失败"
        else:
            kg_status = "未启用知识图谱生成"

        # 触发清理
        cleanup_old_files()

        return answer, audio_file_output, None, kg_status, question_text  # 返回None给kg_image

    except Exception as e:
        print(f"语音问题处理失败: {e}")
        return f"语音处理失败: {str(e)}", None, None, "", ""


def generate_knowledge_graph_optimized(question, answer):
    """生成农业知识图谱 - 优化性能版本，修复字体问题"""
    if not question.strip() or not answer.strip():
        return None, "请先提问并获得答案"

    try:
        # 改进的提示词，确保不同问题生成不同的知识图谱
        kg_prompt = f"""
基于以下问答内容，提取3-5个核心概念及其关系：

问题：{question}
答案：{answer}

请直接返回关系对，格式：概念-关系->概念
最多返回5个关系对，使用中文。
注意：关系对必须基于上述问答内容，确保关系对与问题和答案紧密相关。

关系对示例：
雄蕊-位于->植株顶部
去雄操作-提高->玉米产量
雄蕊识别-需要->图像检测

直接返回关系对，不要有其他解释：
"""

        # 设置更短的超时时间
        relationships_result = [None]

        def get_relationships():
            try:
                relationships_result[0] = agriculture_llm.get_expert_answer(kg_prompt, f"问题: {question}")
            except Exception as e:
                relationships_result[0] = f"AI提取失败: {str(e)}"

        relationships_thread = threading.Thread(target=get_relationships)
        relationships_thread.daemon = True
        relationships_thread.start()
        relationships_thread.join(timeout=8)  # 8秒超时

        if relationships_result[0] is None:
            return generate_fallback_kg(question)

        relationships = relationships_result[0]
        print(f"知识图谱关系提取结果: {relationships}")

        # 创建知识图谱
        width, height = 600, 400
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # 基于问题类型选择预定义的关系对作为备选
        question_lower = question.lower()

        if any(word in question_lower for word in ['去雄', '去除', '操作']):
            predefined_relationships = [
                ("玉米雄蕊", "位于", "植株顶部"),
                ("去雄操作", "提高", "玉米产量"),
                ("雄蕊识别", "需要", "图像检测"),
                ("检测技术", "辅助", "农业管理"),
                ("操作时机", "影响", "去雄效果")
            ]
        elif any(word in question_lower for word in ['病害', '虫害', '防治']):
            predefined_relationships = [
                ("玉米病害", "影响", "雄蕊发育"),
                ("病害防治", "需要", "及时识别"),
                ("病虫害", "导致", "产量下降"),
                ("防治措施", "包括", "农药使用"),
                ("预防措施", "减少", "病害发生")
            ]
        else:
            predefined_relationships = [
                ("玉米雄蕊", "位于", "植株顶部"),
                ("去雄操作", "提高", "玉米产量"),
                ("雄蕊识别", "需要", "图像检测"),
                ("检测技术", "辅助", "农业管理"),
                ("玉米植株", "包含", "雄蕊雌蕊")
            ]

        nodes = set()
        edges = []

        # 解析关系对，如果解析失败使用预定义的
        valid_relationships_found = False
        for line in relationships.split('\n'):
            line = line.strip()
            if '->' in line and '-' in line:
                try:
                    # 简单的解析逻辑
                    if ' - ' in line:
                        left, right = line.split(' -> ')
                        if ' - ' in left:
                            source, relation = left.split(' - ')
                            target = right
                            nodes.add(source.strip())
                            nodes.add(target.strip())
                            edges.append((source.strip(), relation.strip(), target.strip()))
                            valid_relationships_found = True
                except:
                    continue

        # 如果没有有效关系，使用预定义的
        if not valid_relationships_found or len(nodes) < 2:
            nodes = set()
            edges = []
            for source, relation, target in predefined_relationships:
                nodes.add(source)
                nodes.add(target)
                edges.append((source, relation, target))

        # 限制节点数量，提高性能
        nodes = list(nodes)[:6]  # 最多6个节点
        edges = edges[:8]  # 最多8条边

        # 获取字体 - 修复字体问题
        font, small_font = get_fonts()
        if font is None:
            font = ImageFont.load_default()
        if small_font is None:
            small_font = ImageFont.load_default()

        # 简化的绘制逻辑
        center_x, center_y = width // 2, height // 2

        # 简单的网格布局而不是圆形布局
        node_positions = {}
        cols = 3
        node_radius = 40  # 稍微增大节点半径
        spacing_x = width // (cols + 1)
        spacing_y = height // 3

        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols
            x = spacing_x * (col + 1)
            y = spacing_y * (row + 1)
            node_positions[node] = (x, y)

            # 简单的节点绘制
            color = (70, 130, 180) if i % 2 == 0 else (34, 139, 34)
            draw.ellipse([x - node_radius, y - node_radius, x + node_radius, y + node_radius],
                         fill=color, outline='black', width=2)

            # 简化的文字绘制 - 使用正确的字体
            node_text = node[:4]  # 显示前4个字
            try:
                # 计算文本尺寸
                bbox = draw.textbbox((0, 0), node_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text((x - text_width // 2, y - text_height // 2), node_text, fill='white', font=font)
            except:
                # 如果字体渲染失败，使用简单文本
                text_width = len(node_text) * 10
                draw.text((x - text_width // 2, y - 8), node_text, fill='white')

        # 简化的边绘制
        for source, relation, target in edges:
            if source in node_positions and target in node_positions:
                x1, y1 = node_positions[source]
                x2, y2 = node_positions[target]

                # 简单连线
                draw.line([x1, y1, x2, y2], fill=(100, 100, 100), width=2)

                # 简化的箭头
                draw_arrow_simple(draw, x1, y1, x2, y2)

        # 简化的标题 - 使用正确的字体
        try:
            title_bbox = draw.textbbox((0, 0), "知识图谱", font=font)
            title_width = title_bbox[2] - title_bbox[0]
            draw.text((width // 2 - title_width // 2, 15), "知识图谱", fill=(70, 130, 180), font=font)
        except:
            draw.text((width // 2 - 40, 15), "知识图谱", fill=(70, 130, 180))

        draw.text((20, height - 25), f"概念: {len(nodes)} 关系: {len(edges)}", fill=(100, 100, 100))

        # 优化保存
        output_path = os.path.join(tempfile.gettempdir(), f"kg_{uuid.uuid4().hex}.png")
        img.save(output_path, optimize=True, quality=85)

        filename = os.path.basename(output_path)
        performance_stats["generated_files"].add(filename)

        return output_path, f"✅ 知识图谱生成完成 ({len(nodes)}个概念, {len(edges)}条关系)"

    except Exception as e:
        print(f"知识图谱生成失败: {e}")
        # 返回一个极简的备选图谱
        return generate_fallback_kg(question)


def generate_fallback_kg(question):
    """生成极简的备选知识图谱 - 修复字体问题"""
    try:
        width, height = 400, 300
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # 获取字体
        font, small_font = get_fonts()
        if font is None:
            font = ImageFont.load_default()
        if small_font is None:
            small_font = ImageFont.load_default()

        # 根据问题类型选择中心节点
        question_lower = question.lower()
        if any(word in question_lower for word in ['去雄', '去除', '操作']):
            center_text = "去雄"
            concepts = ["雄蕊", "时机", "方法", "效果"]
        elif any(word in question_lower for word in ['病害', '虫害', '防治']):
            center_text = "防治"
            concepts = ["病害", "识别", "农药", "预防"]
        else:
            center_text = "核心"
            concepts = ["雄蕊", "去雄", "检测", "产量"]

        # 中心节点
        center_x, center_y = width // 2, height // 2
        draw.ellipse([center_x - 40, center_y - 40, center_x + 40, center_y + 40],
                     fill=(70, 130, 180), outline='black', width=2)

        try:
            core_bbox = draw.textbbox((0, 0), center_text, font=font)
            core_width = core_bbox[2] - core_bbox[0]
            core_height = core_bbox[3] - core_bbox[1]
            draw.text((center_x - core_width // 2, center_y - core_height // 2), center_text, fill='white', font=font)
        except:
            draw.text((center_x - 20, center_y - 10), center_text, fill='white')

        # 4个方向的基础概念
        positions = [
            (center_x, center_y - 80),  # 上
            (center_x + 80, center_y),  # 右
            (center_x, center_y + 80),  # 下
            (center_x - 80, center_y)  # 左
        ]

        colors = [(34, 139, 34), (46, 139, 87), (139, 69, 19), (72, 61, 139)]

        for i, (concept, pos) in enumerate(zip(concepts, positions)):
            x, y = pos
            draw.ellipse([x - 30, y - 30, x + 30, y + 30],
                         fill=colors[i], outline='black', width=1)

            # 使用正确的字体绘制文字
            try:
                concept_bbox = draw.textbbox((0, 0), concept, font=font)
                concept_width = concept_bbox[2] - concept_bbox[0]
                concept_height = concept_bbox[3] - concept_bbox[1]
                draw.text((x - concept_width // 2, y - concept_height // 2), concept, fill='white', font=font)
            except:
                draw.text((x - 10, y - 8), concept, fill='white')

            # 连线
            draw.line([center_x, center_y, x, y], fill=(100, 100, 100), width=2)
            draw_arrow_simple(draw, center_x, center_y, x, y)

        # 标题
        try:
            title_bbox = draw.textbbox((0, 0), "知识图谱", font=font)
            title_width = title_bbox[2] - title_bbox[0]
            draw.text((width // 2 - title_width // 2, 10), "知识图谱", fill=(70, 130, 180), font=font)
        except:
            draw.text((width // 2 - 30, 10), "知识图谱", fill=(70, 130, 180))

        draw.text((10, height - 20), "简化版本", fill=(150, 150, 150))

        output_path = os.path.join(tempfile.gettempdir(), f"kg_fallback_{uuid.uuid4().hex}.png")
        img.save(output_path, optimize=True)

        filename = os.path.basename(output_path)
        performance_stats["generated_files"].add(filename)

        return output_path, "✅ 基础知识图谱已生成（简化版）"

    except Exception as e:
        print(f"备选知识图谱也失败: {e}")
        return None, f"知识图谱生成失败，请稍后重试"


# 修改问答处理函数，修复语音生成问题
def handle_question_with_context(question, enable_voice_answer, generate_kg):
    """处理文本提问 - 结合检测上下文，修复语音合成"""
    if not question.strip():
        return "请先输入您关于玉米去雄的问题", None, "请先提问并获得答案"

    try:
        # 获取检测上下文
        detection_context = get_detection_context()
        print(f"检测上下文: {detection_context}")

        # 构建智能提示词，结合检测数据
        enhanced_prompt = f"""
        用户提问：{question}

        当前检测数据：{detection_context}

        请直接、专业地回答用户问题，要求：
        - 立即给出答案，不要有任何思考过程、分析步骤或内部推理
        - 基于检测数据提供具体信息
        - 回答要详细但直接相关，避免无关解释
        - 使用清晰的专业语言
        - 如果涉及操作，直接给出步骤，不要解释为什么

        直接回答：
        """

        # 使用增强的提示词获取回答
        answer_response = agriculture_llm.get_expert_answer(enhanced_prompt, question)
        # 清理响应，去除思考过程
        answer = clean_ai_response(answer_response)

        print(f"✅ AI回答生成完成，长度: {len(answer)}")

        # 语音回答 - 使用锁避免并发问题
        audio_file = None
        if enable_voice_answer and answer:
            print("🔊 开始语音合成...")
            try:
                speech_processor = get_speech_processor()
                if speech_processor and hasattr(speech_processor, 'tts_engine') and speech_processor.tts_engine:
                    # 使用锁确保同一时间只有一个语音生成任务
                    with voice_lock:
                        audio_result = [None]

                        # 在 handle_question_with_context 和 handle_voice_question_with_context 中
                        # 修改语音合成部分：

                        def generate_audio():
                            try:
                                speech_processor = get_speech_processor()
                                # 使用修复后的 text_to_speech 方法
                                audio_file_path = speech_processor.text_to_speech(answer, save_to_file=True)
                                if audio_file_path and os.path.exists(audio_file_path):
                                    filename = os.path.basename(audio_file_path)
                                    performance_stats["generated_files"].add(filename)
                                    audio_result[0] = audio_file_path
                                    print(f"✅ 语音文件生成成功: {audio_file_path}")
                                else:
                                    print("❌ 语音文件生成失败")
                            except Exception as e:
                                print(f"❌ 语音生成线程异常: {e}")

                        audio_thread = threading.Thread(target=generate_audio)
                        audio_thread.daemon = True
                        audio_thread.start()
                        audio_thread.join(timeout=15)

                        audio_file = audio_result[0]

                    if not audio_file:
                        print("⚠️ 语音生成超时或失败")
                else:
                    print("❌ 语音合成引擎不可用")
            except Exception as e:
                print(f"❌ 语音合成处理失败: {e}")
        else:
            print("🔇 语音回答未启用")

        # 知识图谱生成 - 修改为保存到历史记录
        kg_status = ""
        if generate_kg and answer:
            try:
                kg_image, kg_status = generate_knowledge_graph_optimized(question, answer)
                if kg_image:
                    # 保存到知识图谱历史记录
                    save_knowledge_graph_record(question, answer, kg_image)
                    kg_status = "✅ 知识图谱已生成并保存到知识图谱库"
                else:
                    kg_status = "❌ 知识图谱生成失败"
            except Exception as e:
                print(f"知识图谱生成失败: {e}")
                kg_status = "知识图谱生成失败"
        else:
            kg_status = "未启用知识图谱生成"

        # 触发清理
        cleanup_old_files()

        return answer, audio_file, kg_status  # 返回None给kg_image，因为不在当前界面显示

    except Exception as e:
        print(f"问答处理失败: {e}")
        error_msg = f"AI服务暂时不可用，请稍后重试。错误: {str(e)}"
        return error_msg, None, "知识图谱生成失败"



def generate_comprehensive_report(area_info="", date_range="全部数据"):
    """生成综合工作报告 - 优化性能"""
    if not detection_history:
        return "暂无检测数据，请先进行检测操作", None

    try:
        # 筛选数据
        relevant_data = [
            record for record in detection_history
            if not area_info or area_info in record.get('area', '')
        ]

        if not relevant_data:
            return f"区域 '{area_info}' 暂无检测数据", None

        # 计算统计数据
        total_detections = len(relevant_data)
        total_stamens = sum(record['data'].get('total_count', 0) for record in relevant_data)
        avg_stamens = total_stamens / total_detections if total_detections > 0 else 0

        # 按区域统计
        area_stats = {}
        for record in relevant_data:
            area = record.get('area', '未分类')
            if area not in area_stats:
                area_stats[area] = {'count': 0, 'stamens': 0}
            area_stats[area]['count'] += 1
            area_stats[area]['stamens'] += record['data'].get('total_count', 0)

        # 生成详细统计信息
        stats_text = f"""
## 📊 综合检测报告

### 总体统计
- **总检测次数**: {total_detections} 次
- **总雄蕊数量**: {total_stamens} 个
- **平均每张图像**: {avg_stamens:.1f} 个雄蕊
- **时间范围**: {date_range}
- **区域筛选**: {area_info if area_info else "全部区域"}

### 区域统计
"""
        for area, stats in area_stats.items():
            stats_text += f"- **{area}**: {stats['count']} 次检测，{stats['stamens']} 个雄蕊\n"

        # 使用LLM生成智能分析报告 - 使用线程避免阻塞
        analysis_report = ""

        def generate_analysis_thread():
            nonlocal analysis_report
            try:
                analysis_prompt = f"""
基于以下玉米去雄检测数据，生成一份专业的综合分析报告：

**数据概况**：
- 总检测次数: {total_detections}
- 总雄蕊数量: {total_stamens}
- 平均每张图像雄蕊数: {avg_stamens:.1f}
- 区域分布: {list(area_stats.keys())}

请从以下角度进行分析：
1. 工作完成情况评估
2. 雄蕊分布特点分析  
3. 可能存在的问题和改进建议
4. 下一步工作计划

要求：
- 直接给出分析结果，不要有任何思考过程或解释
- 分点列出，清晰明了
- 基于数据给出具体建议
- 使用通俗易懂的语言

直接给出分析报告：
"""
                analysis_response = agriculture_llm.get_expert_answer(analysis_prompt, f"检测次数: {total_detections}")
                analysis_report = clean_ai_response(analysis_response)
            except Exception as e:
                print(f"生成分析报告失败: {e}")
                analysis_report = "专业分析生成失败，请稍后重试"

        # 启动分析线程
        analysis_thread = threading.Thread(target=generate_analysis_thread)
        analysis_thread.daemon = True
        analysis_thread.start()
        analysis_thread.join(timeout=15)  # 15秒超时

        if analysis_thread.is_alive():
            analysis_report = "专业分析生成超时，请稍后重试"

        full_report = stats_text + f"\n### 📋 专业分析\n{analysis_report}"

        # 生成PDF报告 - 使用线程避免阻塞
        pdf_path = None

        def generate_pdf_thread():
            nonlocal pdf_path
            try:
                pdf_path = generate_pdf_report(full_report, area_info, date_range, total_detections, total_stamens, area_stats, analysis_report)
            except Exception as e:
                print(f"生成PDF失败: {e}")

        pdf_thread = threading.Thread(target=generate_pdf_thread)
        pdf_thread.daemon = True
        pdf_thread.start()
        pdf_thread.join(timeout=10)  # 10秒超时

        # 触发清理
        cleanup_old_files()

        return full_report, pdf_path

    except Exception as e:
        return f"生成报告时出错: {str(e)}", None

def generate_pdf_report(report_content, area_info, date_range, total_detections, total_stamens, area_stats, analysis_report):
    """生成PDF格式的报告 - 优化性能"""
    try:
        pdf_path = os.path.join(tempfile.gettempdir(), f"corn_detection_report_{uuid.uuid4().hex}.pdf")

        # 方法1：使用reportlab（如果可用）
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont

            # 尝试注册中文字体
            try:
                font_paths = [
                    "C:/Windows/Fonts/simsun.ttc",  # Windows 宋体
                    "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
                    "/System/Library/Fonts/PingFang.ttc",  # macOS
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
                ]

                chinese_font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        try:
                            font_name = os.path.basename(font_path).split('.')[0]
                            pdfmetrics.registerFont(TTFont(font_name, font_path))
                            chinese_font = font_name
                            break
                        except:
                            continue

                if not chinese_font:
                    chinese_font = "Helvetica"
            except:
                chinese_font = "Helvetica"

            doc = SimpleDocTemplate(pdf_path, pagesize=A4)

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName=chinese_font,
                fontSize=16,
                spaceAfter=30,
                alignment=1
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontName=chinese_font,
                fontSize=14
            )
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontName=chinese_font,
                fontSize=10
            )

            story = []

            title = Paragraph("玉米去雄检测工作报告", title_style)
            story.append(title)
            story.append(Spacer(1, 0.2 * inch))

            info_text = f"<b>生成时间:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>"
            info_text += f"<b>区域筛选:</b> {area_info if area_info else '全部区域'}<br/>"
            info_text += f"<b>时间范围:</b> {date_range}<br/>"
            info_text += f"<b>总检测次数:</b> {total_detections}<br/>"
            info_text += f"<b>总雄蕊数量:</b> {total_stamens}<br/>"

            info_para = Paragraph(info_text, normal_style)
            story.append(info_para)
            story.append(Spacer(1, 0.3 * inch))

            if area_stats:
                area_data = [['区域', '检测次数', '雄蕊数量']]
                for area, stats in area_stats.items():
                    area_data.append([area, str(stats['count']), str(stats['stamens'])])

                table = Table(area_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), chinese_font),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('FONTNAME', (0, 1), (-1, -1), chinese_font),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 0.3 * inch))

            if analysis_report:
                clean_analysis = analysis_report.replace('**', '').replace('###', '').strip()
                analysis_para = Paragraph(f"<b>专业分析</b><br/>{clean_analysis}", normal_style)
                story.append(analysis_para)

            doc.build(story)

            filename = os.path.basename(pdf_path)
            performance_stats["generated_files"].add(filename)

            return pdf_path

        except ImportError:
            print("⚠️ reportlab不可用，生成文本格式报告")
            txt_path = os.path.join(tempfile.gettempdir(), f"corn_detection_report_{uuid.uuid4().hex}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("玉米去雄检测工作报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"区域筛选: {area_info if area_info else '全部区域'}\n")
                f.write(f"时间范围: {date_range}\n")
                f.write(f"总检测次数: {total_detections}\n")
                f.write(f"总雄蕊数量: {total_stamens}\n\n")

                if area_stats:
                    f.write("区域统计:\n")
                    for area, stats in area_stats.items():
                        f.write(f"  {area}: {stats['count']}次检测, {stats['stamens']}个雄蕊\n")
                    f.write("\n")

                if analysis_report:
                    f.write("专业分析:\n")
                    clean_analysis = analysis_report.replace('**', '').replace('###', '').strip()
                    f.write(clean_analysis + "\n")

            filename = os.path.basename(txt_path)
            performance_stats["generated_files"].add(filename)

            return txt_path

    except Exception as e:
        print(f"生成PDF失败: {e}")
        return None


def fast_inference_with_brightness(img, threshold, brightness_factor, area_info, enable_guidance):
    """带亮度调节的快速检测推理函数"""
    processed_img = preprocess_image(img)
    if processed_img is None:
        return None, "请先上传有效的图像文件", "", None

    try:
        # 记录开始时间
        start_time = datetime.now()

        # 执行带亮度调节的检测
        from Stamen_detection import detect_stamen_with_brightness
        boxed_pil, detection_result, histogram_img = detect_stamen_with_brightness(
            processed_img, threshold, brightness_factor
        )

        # 计算处理时间
        process_time = (datetime.now() - start_time).total_seconds()

        # 生成检测结果文本
        detection_text = detection_result.to_text_summary()

        # 添加位置信息
        if hasattr(detection_result, 'to_position_summary'):
            position_text = detection_result.to_position_summary()
            full_detection_text = f"{detection_text}\n\n{position_text}"
        else:
            full_detection_text = detection_text

        # 添加处理时间信息
        full_detection_text = f"⏱️ 处理时间: {process_time:.2f}秒\n亮度调节: {brightness_factor}x\n\n{full_detection_text}"

        # 保存检测记录
        detection_data = detection_result.get_detection_data()
        save_detection_record(detection_data, area_info)

        # 保存亮度直方图
        histogram_path = None
        if histogram_img:
            histogram_path = os.path.join(tempfile.gettempdir(), f"histogram_{uuid.uuid4().hex}.png")
            histogram_img.save(histogram_path)
            performance_stats["generated_files"].add(os.path.basename(histogram_path))

        # 生成操作指导
        guidance_text = ""
        if enable_guidance and detection_result.detected_objects:
            try:
                guidance_prompt = f"""
检测结果：{detection_text}
亮度调节：{brightness_factor}x

请直接给出具体的去雄操作指导，要求：
- 直接给出操作步骤，不要有任何思考过程
- 步骤清晰实用，易于执行
- 基于检测结果针对性建议
- 使用通俗易懂的语言
- 不超过5个步骤

操作指导：
"""
                guidance_response = agriculture_llm.get_expert_answer(guidance_prompt, detection_text)
                guidance_text = clean_ai_response(guidance_response)
            except Exception as e:
                print(f"生成指导失败: {e}")
                guidance_text = "AI指导暂时不可用，请参考标准操作流程。"

        # 保存结果图像
        out_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.jpg")
        out_path = safe_image_save(boxed_pil, out_path, quality=85)

        print(
            f"带亮度调节检测完成，亮度: {brightness_factor}x，发现 {len(detection_result.detected_objects)} 个雄蕊，耗时 {process_time:.2f}秒")

        # 调试信息
        if detection_result.detected_objects:
            for i, obj in enumerate(detection_result.detected_objects):
                print(f"检测对象 {i}: 标签={obj['label']}, 置信度={obj['confidence']:.2f}, 位置={obj['position']}")

        return out_path, full_detection_text, guidance_text, histogram_path

    except Exception as e:
        error_msg = f"检测失败: {str(e)}"
        print(f"错误详情: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈
        return None, error_msg, "", None


def clear_history():
    """清空检测历史"""
    global detection_history
    detection_history = []

    # 强制垃圾回收
    gc.collect()

    return "检测历史已清空", None


# 创建Gradio界面
with gr.Blocks(
        title=TITLE,
        theme=gr.themes.Soft(
            primary_hue="gray",
            secondary_hue="gray",
            neutral_hue="gray"
        ),
        css="""
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }
    .header-section {
        text-align: center;
        margin-bottom: 30px;
        padding: 25px;
        background: white;
        border-radius: 8px;
        border: 2px solid #000;
        color: #000;
    }
    .tab-content {
        padding: 20px;
        background: white;
        border-radius: 8px;
        border: 1px solid #000;
        margin-bottom: 20px;
    }
    .input-section {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #000;
    }
    .result-section {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #000;
    }
    .stats-card {
        background: white;
        color: #000;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
        border: 1px solid #000;
    }
    .voice-card {
        background: white;
        color: #000;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
        border: 1px solid #000;
    }
    .detection-card {
        background: white;
        color: #000;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
        border: 1px solid #000;
    }
    .btn-primary {
        background: #000;
        border: 1px solid #000;
        color: white;
        font-weight: 500;
    }
    .btn-primary:hover {
        background: #333;
        border: 1px solid #333;
    }
    .btn-secondary {
        background: #666;
        border: 1px solid #666;
        color: white;
        font-weight: 500;
    }
    .btn-secondary:hover {
        background: #888;
        border: 1px solid #888;
    }
    .gradio-image {
        border-radius: 6px;
        border: 1px solid #000;
    }
    .gradio-textbox textarea {
        border-radius: 6px;
        border: 1px solid #000;
        background: white;
    }
    .gradio-slider {
        background: white;
    }
    .accordion-header {
        background: #f5f5f5 !important;
        border: 1px solid #000 !important;
    }
    .knowledge-graph-card {
        background: white;
        color: #000;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
        border: 1px solid #000;
    }
    .pdf-download-card {
        background: white;
        color: #000;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
        border: 1px solid #000;
    }
    .kg-gallery-card {
        background: white;
        color: #000;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
        border: 1px solid #000;
    }
    """) as demo:
    # 头部区域
    with gr.Column(elem_classes=["header-section"]):
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(f"### {DESCRIPTION}")

    with gr.Column(elem_classes=["main-container"]):
        with gr.Tab("🔍 图像检测"):
            # ... 图像检测Tab内容保持不变 ...
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["input-section"]):
                        gr.Markdown("### 📸 图像输入")
                        in_img = gr.Image(
                            label="上传玉米植株图像",
                            type="filepath",
                            sources=["upload", "webcam"],
                            height=280
                        )

                        with gr.Row():
                            threshold = gr.Slider(
                                0.1, 0.95, 0.5, step=0.05,
                                label="检测灵敏度",
                                info="值越高，检测越严格"
                            )

                        with gr.Row():
                            brightness_slider = gr.Slider(
                                0.1, 3.0, 1.0, step=0.1,
                                label="图像亮度调节",
                                info="1.0为原始亮度，小于1调暗，大于1调亮"
                            )

                        with gr.Row():
                            area_info = gr.Textbox(
                                label="工作区域",
                                placeholder="例如：A01",
                                max_lines=1
                            )

                        with gr.Row():
                            enable_guidance = gr.Checkbox(
                                label="启用AI操作指导",
                                value=True
                            )

                            btn_detect = gr.Button(
                                "开始检测",
                                variant="primary",
                                size="lg"
                            )

                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["result-section"]):
                        gr.Markdown("### 📊 检测结果")

                        with gr.Accordion("📈 图像亮度分析", open=True):
                            brightness_histogram = gr.Image(
                                label="亮度分布直方图",
                                type="filepath",
                                height=200,
                                show_download_button=True
                            )

                        out_img = gr.Image(
                            label="检测结果",
                            type="filepath",
                            height=300,
                            show_download_button=True
                        )

                        with gr.Group(elem_classes=["stats-card"]):
                            stats_display = gr.Textbox(
                                label="实时统计",
                                value="等待检测...",
                                lines=2,
                                interactive=False
                            )

                        with gr.Accordion("检测详情", open=True):
                            out_detection = gr.Textbox(
                                label="",
                                lines=4,
                                show_copy_button=True
                            )

                        with gr.Accordion("操作指导", open=True):
                            out_guidance = gr.Textbox(
                                label="",
                                lines=3,
                                show_copy_button=True
                            )

        with gr.Tab("💬 智能问答"):
            # 修改智能问答界面，移除知识图谱显示区域
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["input-section"]):
                        gr.Markdown("### 🎤 语音提问")

                        with gr.Group(elem_classes=["stats-card"]):
                            detection_status = gr.Textbox(
                                label="检测数据状态",
                                value="等待检测数据...",
                                lines=2,
                                interactive=False
                            )
                            btn_refresh_status = gr.Button("刷新检测状态", variant="secondary", size="sm")

                        with gr.Group(elem_classes=["voice-card"]):
                            voice_input = gr.Audio(
                                label="录制语音问题",
                                sources=["microphone"],
                                type="filepath"
                            )
                            btn_voice_question = gr.Button("识别语音并提问", variant="primary")

                        gr.Markdown("### 📝 文本提问")

                        preset_questions = [
                            "当前雄蕊数量多少？",
                            "雄蕊分布位置如何？",
                            "检测结果可信吗？",
                            "接下来该做什么操作？",
                            "如何提高检测准确性？",
                            "雄蕊数量正常吗？",
                            "需要去雄操作吗？",
                            "检测到的雄蕊质量如何？"
                        ]

                        with gr.Accordion("💡 常用问题示例", open=False):
                            gr.Markdown("点击以下问题快速提问：")
                            with gr.Row():
                                preset_buttons = []
                                for i, question in enumerate(preset_questions[:4]):
                                    btn = gr.Button(question, size="sm")
                                    preset_buttons.append(btn)
                            with gr.Row():
                                for i, question in enumerate(preset_questions[4:]):
                                    btn = gr.Button(question, size="sm")
                                    preset_buttons.append(btn)

                        question_input = gr.Textbox(
                            label="输入问题",
                            placeholder="例如：当前雄蕊数量多少？如何防治病虫害？雄蕊位置分布如何？",
                            lines=3
                        )

                        with gr.Row():
                            enable_voice_answer = gr.Checkbox(
                                label="启用语音回答",
                                value=True
                            )
                            generate_kg = gr.Checkbox(
                                label="生成知识图谱",
                                value=True,
                                info="图谱将保存到知识图谱库"
                            )

                        btn_question = gr.Button("发送问题", variant="primary", size="lg")

                        voice_question_text = gr.Textbox(
                            label="语音识别结果",
                            lines=2,
                            interactive=False,
                            show_copy_button=True
                        )

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["result-section"]):
                        # 移除知识图谱显示区域，只显示专家解答
                        gr.Markdown("### 💡 专家解答")
                        answer_output = gr.Textbox(
                            label="AI回答",
                            lines=12,
                            show_copy_button=True
                        )

                        voice_answer_audio = gr.Audio(
                            label="语音回答",
                            type="filepath",
                            show_download_button=True
                        )

                        # 只显示知识图谱生成状态
                        kg_status = gr.Textbox(
                            label="知识图谱状态",
                            lines=2,
                            interactive=False
                        )

        with gr.Tab("🧠 知识图谱库"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["kg-gallery-card"]):
                        gr.Markdown("### 📚 知识图谱画廊")
                        kg_gallery = gr.Gallery(
                            label="历史知识图谱",
                            columns=4,
                            rows=2,
                            height=400,
                            object_fit="contain",
                            show_label=False
                        )
                        gallery_status = gr.Textbox(
                            label="画廊状态",
                            lines=1,
                            interactive=False
                        )

                        with gr.Row():
                            btn_refresh_gallery = gr.Button("刷新画廊", variant="secondary")
                            btn_clear_kg_history = gr.Button("清空历史", variant="secondary")

                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["result-section"]):
                        gr.Markdown("### 🔍 图谱详情")
                        selected_kg_image = gr.Image(
                            label="选中的知识图谱",
                            type="filepath",
                            height=300,
                            show_download_button=True
                        )

                        selected_question = gr.Textbox(
                            label="问题",
                            lines=2,
                            interactive=False
                        )

                        selected_answer = gr.Textbox(
                            label="答案摘要",
                            lines=4,
                            interactive=False
                        )

                        selected_timestamp = gr.Textbox(
                            label="生成时间",
                            lines=1,
                            interactive=False
                        )

        with gr.Tab("📈 工作报告"):
            # ... 工作报告Tab内容保持不变 ...
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["input-section"]):
                        gr.Markdown("### 📋 报告设置")
                        report_area = gr.Textbox(
                            label="区域筛选",
                            placeholder="留空则统计所有区域"
                        )
                        date_filter = gr.Dropdown(
                            choices=["今日", "最近3天", "最近7天", "全部数据"],
                            label="时间范围",
                            value="全部数据"
                        )
                        with gr.Row():
                            btn_report = gr.Button("生成综合报告", variant="primary")
                            btn_clear = gr.Button("清空历史", variant="secondary")

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["result-section"]):
                        gr.Markdown("### 📄 综合检测报告")
                        report_output = gr.Markdown(
                            label="",
                            value="点击'生成综合报告'查看统计数据和分析"
                        )

                        with gr.Group(elem_classes=["pdf-download-card"]):
                            gr.Markdown("### 📥 报告下载")
                            pdf_download = gr.File(
                                label="下载检测报告",
                                file_types=[".pdf", ".txt"],
                                visible=False
                            )


    # 绑定事件处理
    def update_stats_with_brightness(img, threshold, brightness_factor, area_info, enable_guidance):
        """更新统计信息和检测状态（带亮度调节）"""
        result = fast_inference_with_brightness(img, threshold, brightness_factor, area_info, enable_guidance)
        if result[0] is not None:
            detection_text = result[1]
            if "检测到" in detection_text:
                stats = detection_text.split('\n')[0]
                status = update_detection_status()
                return result + (stats, status)
        status = update_detection_status()
        return result + ("等待检测...", status)


    # 确保使用正确的检测函数
    btn_detect.click(
        update_stats_with_brightness,
        inputs=[in_img, threshold, brightness_slider, area_info, enable_guidance],
        outputs=[out_img, out_detection, out_guidance, brightness_histogram, stats_display, detection_status]
    )


    btn_question.click(
        handle_question_with_context,
        inputs=[question_input, enable_voice_answer, generate_kg],
        outputs=[answer_output, voice_answer_audio, kg_status]
    )

    btn_voice_question.click(
        handle_voice_question_with_context,
        inputs=[voice_input, enable_voice_answer, generate_kg],
        outputs=[answer_output, voice_answer_audio, kg_status, voice_question_text]
    )

    btn_refresh_status.click(
        update_detection_status,
        outputs=[detection_status]
    )


    # 知识图谱库事件处理
    def refresh_kg_gallery():
        """刷新知识图谱画廊"""
        gallery_data, status = get_knowledge_graph_gallery()
        return gallery_data, status


    def clear_kg_history():
        """清空知识图谱历史"""
        global knowledge_graph_history
        # 删除所有知识图谱文件
        for record in knowledge_graph_history:
            try:
                if os.path.exists(record['kg_image']):
                    os.remove(record['kg_image'])
            except:
                pass
        knowledge_graph_history = []
        return [], "知识图谱历史已清空"


    def on_gallery_select(evt: gr.SelectData):
        """处理画廊选择事件"""
        if evt.index is not None and knowledge_graph_history:
            # 获取选中的知识图谱记录（画廊是倒序显示的）
            reversed_index = len(knowledge_graph_history) - 1 - evt.index
            if 0 <= reversed_index < len(knowledge_graph_history):
                record = knowledge_graph_history[reversed_index]
                question, answer, timestamp = get_knowledge_graph_detail(record['kg_image'])
                return record['kg_image'], question, answer, timestamp
        return None, "请选择知识图谱", "", ""


    # 绑定知识图谱库事件
    btn_refresh_gallery.click(
        refresh_kg_gallery,
        outputs=[kg_gallery, gallery_status]
    )

    btn_clear_kg_history.click(
        clear_kg_history,
        outputs=[kg_gallery, gallery_status]
    )

    kg_gallery.select(
        on_gallery_select,
        outputs=[selected_kg_image, selected_question, selected_answer, selected_timestamp]
    )

    # 初始化时刷新画廊
    demo.load(
        refresh_kg_gallery,
        outputs=[kg_gallery, gallery_status]
    )

    for btn in preset_buttons:
        btn.click(
            lambda q=btn.value: q,
            outputs=[question_input]
        )
    for btn in preset_buttons:
        btn.click(
            lambda q=btn.value: q,
            outputs=[question_input]
        )


    def generate_report_with_pdf(area_info, date_range):
        """生成报告并创建PDF"""
        report, pdf_path = generate_comprehensive_report(area_info, date_range)
        if pdf_path:
            return report, gr.File(value=pdf_path, visible=True)
        else:
            return report, gr.File(visible=False)


    btn_report.click(
        generate_report_with_pdf,
        inputs=[report_area, date_filter],
        outputs=[report_output, pdf_download]
    )


    def clear_history_with_update():
        """清空历史并更新界面"""
        msg, _ = clear_history()
        return msg, gr.File(visible=False)


    btn_clear.click(
        clear_history_with_update,
        outputs=[report_output, pdf_download]
    )

if __name__ == "__main__":
    try:
        print("🚀 启动智能农业助手系统...")
        print("📱 访问地址: http://localhost:7860")
        print("🧠 知识图谱功能已启用")
        print("📄 PDF报告功能已启用（支持中文）")
        print("🎤 语音交互问答功能已启用")
        print("📊 检测数据上下文功能已集成")
        print("🧹 AI思考过程清理功能已启用")
        print("⚡ 性能优化已启用（线程、缓存、清理）")

        # 预加载字体
        get_fonts()

        # 启动时清理
        cleanup_old_files()

        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            debug=False,
            quiet=True,
            prevent_thread_lock=False
        )
    except Exception as e:
        print(f"启动失败: {e}")
