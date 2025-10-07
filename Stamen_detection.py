import base64
import os

import requests
import io
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import time
import socket

API_URL = "https://cb44u2yd67eeu0n9.aistudio-hub.baidu.com/object-detection"
TOKEN = "7b3117b8126a3cf63eb03eaec74e9d3e5969e464"

MAX_RETRIES = 5
RETRY_DELAY = 3


class DetectionResult:
    def __init__(self, raw_response, image_size, threshold=0.5):
        self.raw_response = raw_response
        self.image_size = image_size
        self.threshold = threshold
        self.detected_objects = []
        self._parse_detections()

    def debug_font_system():
        """调试字体系统"""
        print("🔍 调试字体系统...")

        # 检查常见字体路径
        font_paths = [
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        ]

        available_fonts = []
        for font_path in font_paths:
            if os.path.exists(font_path):
                available_fonts.append(font_path)
                print(f"✅ 字体存在: {font_path}")
            else:
                print(f"❌ 字体不存在: {font_path}")

        print(f"可用字体数量: {len(available_fonts)}")
        return available_fonts

    # 在模块加载时调试字体
    try:
        debug_font_system()
    except Exception as e:
        print(f"字体调试失败: {e}")

    def _translate_label(self, label):
        """将标签翻译为中文"""
        if label is None:
            return "未知"

        label_lower = str(label).lower()
        if any(word in label_lower for word in ['stamen', '雄蕊', 'male']):
            return "雄蕊"
        elif any(word in label_lower for word in ['pistil', '雌蕊', 'female']):
            return "雌蕊"
        else:
            return str(label)

    def _parse_detections(self):
        """解析检测结果 - 修复数据结构问题"""
        try:
            print(f"原始响应类型: {type(self.raw_response)}")
            print(f"原始响应内容: {self.raw_response}")

            objects = []

            # 处理不同的响应格式
            if isinstance(self.raw_response, dict):
                # 格式1: {'result': {'detectedObjects': [...]}}
                if 'result' in self.raw_response and isinstance(self.raw_response['result'], dict):
                    objects = self.raw_response['result'].get('detectedObjects', [])
                # 格式2: {'detectedObjects': [...]}
                elif 'detectedObjects' in self.raw_response:
                    objects = self.raw_response['detectedObjects']
                # 格式3: 直接包含对象列表的字典
                elif any(key in self.raw_response for key in ['objects', 'predictions', 'detections']):
                    for key in ['objects', 'predictions', 'detections']:
                        if key in self.raw_response:
                            objects = self.raw_response[key]
                            break
                else:
                    # 如果字典没有预期的键，尝试将其视为单个检测对象
                    objects = [self.raw_response]

            elif isinstance(self.raw_response, list):
                # 直接是对象列表
                objects = self.raw_response
            else:
                print(f"未知的响应格式: {type(self.raw_response)}")
                objects = []

            print(f"解析出的对象数量: {len(objects)}")

            for i, obj in enumerate(objects):
                try:
                    # 确保对象是字典类型
                    if not isinstance(obj, dict):
                        print(f"跳过非字典对象: {type(obj)} - {obj}")
                        continue

                    # 获取置信度 - 处理不同的字段名
                    confidence = 0
                    for score_key in ['score', 'confidence', 'conf']:
                        if score_key in obj:
                            try:
                                confidence = float(obj[score_key])
                                break
                            except (ValueError, TypeError):
                                continue

                    if confidence < self.threshold:
                        continue

                    # 获取边界框 - 处理不同的字段名
                    bbox = {}
                    for bbox_key in ['bbox', 'position', 'box', 'bounding_box']:
                        if bbox_key in obj and obj[bbox_key]:
                            bbox = obj[bbox_key]
                            break

                    # 如果bbox是列表格式 [x, y, w, h]
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        bbox = {
                            'x': bbox[0],
                            'y': bbox[1],
                            'width': bbox[2],
                            'height': bbox[3]
                        }

                    # 获取标签
                    label = None
                    for label_key in ['label', 'class', 'name']:
                        if label_key in obj:
                            label = obj[label_key]
                            break

                    # 使用翻译后的标签
                    translated_label = self._translate_label(label)

                    detection = {
                        'id': i,
                        'label': translated_label,
                        'confidence': confidence,
                        'position': bbox,
                        'status': self._determine_status(obj, confidence),
                        'size': self._calculate_size(bbox)
                    }
                    self.detected_objects.append(detection)
                    print(f"成功解析对象 {i}: {detection}")

                except Exception as e:
                    print(f"解析单个检测对象时出错: {e}")
                    print(f"问题对象: {obj}")
                    continue

        except Exception as e:
            print(f"解析检测结果时出错: {e}")
            self.detected_objects = []

    def _determine_status(self, obj, confidence):
        """判断雄蕊状态"""
        if confidence < 0.5:
            return "不确定"

        # 尝试从对象中获取状态信息
        status = obj.get('status', '')
        label = str(obj.get('label', '')).lower()

        if status:
            return status
        elif any(word in label for word in ['small', 'closed', '未开放']):
            return "未开放"
        elif any(word in label for word in ['large', 'open', '已开放']):
            return "已开放"
        else:
            return "正常"

    def _calculate_size(self, bbox):
        """计算雄蕊大小"""
        if not bbox:
            return 0

        try:
            if isinstance(bbox, dict):
                width = bbox.get('width', bbox.get('w', 0))
                height = bbox.get('height', bbox.get('h', 0))
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                width = bbox[2]
                height = bbox[3]
            else:
                return 0

            return round(float(width) * float(height) / 1000, 2) if width and height else 0
        except (ValueError, TypeError):
            return 0

    def to_text_summary(self):
        """生成文本摘要"""
        total = len(self.detected_objects)
        if total == 0:
            return f"未检测到置信度高于{self.threshold}的雄蕊"

        open_count = sum(1 for obj in self.detected_objects if obj['status'] == '已开放')
        closed_count = sum(1 for obj in self.detected_objects if obj['status'] == '未开放')

        summary = f"检测到{total}个雄蕊（置信度≥{self.threshold}）："
        if closed_count > 0:
            summary += f"未开放{closed_count}个"
        if open_count > 0:
            summary += f"，已开放{open_count}个"

        if self.detected_objects:
            avg_confidence = sum(obj['confidence'] for obj in self.detected_objects) / total
            summary += f"。平均置信度：{avg_confidence:.2f}"

        return summary

    def to_position_summary(self):
        """生成位置信息摘要"""
        if not self.detected_objects:
            return "未检测到雄蕊，无位置信息"

        summary = "雄蕊位置信息：\n"
        for i, obj in enumerate(self.detected_objects):
            pos = obj['position']

            # 处理不同的边界框格式
            if isinstance(pos, dict):
                x = pos.get('x', pos.get('left', '未知'))
                y = pos.get('y', pos.get('top', '未知'))
                w = pos.get('width', pos.get('w', '未知'))
                h = pos.get('height', pos.get('h', '未知'))
            elif isinstance(pos, (list, tuple)) and len(pos) >= 4:
                x, y, w, h = pos[0], pos[1], pos[2], pos[3]
            else:
                x, y, w, h = '未知', '未知', '未知', '未知'

            summary += f"雄蕊{i + 1}: 位置(x={x}, y={y}), 尺寸({w}×{h}), 置信度{obj['confidence']:.2f}, 状态:{obj['status']}\n"

        return summary

    def get_detection_data(self):
        """获取结构化检测数据"""
        return {
            'timestamp': datetime.now().isoformat(),
            'threshold': self.threshold,
            'total_count': len(self.detected_objects),
            'open_count': sum(1 for obj in self.detected_objects if obj['status'] == '已开放'),
            'closed_count': sum(1 for obj in self.detected_objects if obj['status'] == '未开放'),
            'objects': [
                {
                    'label': obj['label'],
                    'confidence': obj['confidence'],
                    'status': obj['status'],
                    'size': obj['size'],
                    'position': obj['position']
                }
                for obj in self.detected_objects
            ]
        }


def resize_image_for_detection(pil_img, max_dimension=1200):
    """调整图像尺寸以优化传输"""
    if max_dimension is None:
        return pil_img

    width, height = pil_img.size
    if width <= max_dimension and height <= max_dimension:
        return pil_img

    if width > height:
        new_width = max_dimension
        new_height = int(height * max_dimension / width)
    else:
        new_height = max_dimension
        new_width = int(width * max_dimension / height)

    return pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def draw_detection_boxes(image, detections):
    """在图像上绘制检测框 - 彻底修复字体问题"""
    if not detections:
        return image

    draw = ImageDraw.Draw(image)

    # 改进字体加载逻辑，确保能够加载中文字体
    font = None
    chinese_font_paths = [
        "C:/Windows/Fonts/simsun.ttc",  # Windows 宋体
        "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
        "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
        "/System/Library/Fonts/PingFang.ttc",  # macOS
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux 备选
    ]

    # 首先尝试加载中文字体
    for font_path in chinese_font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 20)  # 增大字体大小
                print(f"✅ 检测框字体加载成功: {font_path}")
                break
        except Exception as e:
            print(f"字体加载失败 {font_path}: {e}")
            continue

    # 如果中文字体都失败，尝试系统默认字体
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            print("✅ 使用英文字体 Arial")
        except:
            try:
                font = ImageFont.load_default()
                print("⚠️ 使用默认字体（可能不支持中文）")
            except:
                print("❌ 所有字体加载失败，将无法显示文字")
                font = None

    for obj in detections:
        pos = obj['position']
        if not pos:
            continue

        # 处理不同的边界框格式
        if isinstance(pos, dict):
            x = pos.get('x', pos.get('left', 0))
            y = pos.get('y', pos.get('top', 0))
            w = pos.get('width', pos.get('w', 0))
            h = pos.get('height', pos.get('h', 0))
        elif isinstance(pos, (list, tuple)) and len(pos) >= 4:
            x, y, w, h = pos[0], pos[1], pos[2], pos[3]
        else:
            continue

        try:
            x, y, w, h = float(x), float(y), float(w), float(h)
        except (ValueError, TypeError):
            continue

        # 调整框的大小
        scale_factor = 0.9
        center_x, center_y = x + w / 2, y + h / 2
        w_new, h_new = w * scale_factor, h * scale_factor
        x_new, y_new = center_x - w_new / 2, center_y - h_new / 2

        # 确保坐标在图像范围内
        img_width, img_height = image.size
        x_new = max(0, min(x_new, img_width - 10))
        y_new = max(0, min(y_new, img_height - 10))
        w_new = min(w_new, img_width - x_new - 5)
        h_new = min(h_new, img_height - y_new - 5)

        # 根据置信度选择颜色
        confidence = obj['confidence']
        if confidence >= 0.8:
            color = "red"
            border_width = 3
        elif confidence >= 0.6:
            color = "orange"
            border_width = 2
        else:
            color = "yellow"
            border_width = 2

        # 绘制边界框
        draw.rectangle([x_new, y_new, x_new + w_new, y_new + h_new],
                       outline=color, width=border_width)

        # 添加标签和置信度 - 使用更简单的文本渲染方法
        label_text = f"雄蕊 {confidence:.2f}"

        # 计算文本大小 - 使用更简单的方法
        if font:
            try:
                # 使用更可靠的方法获取文本尺寸
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                # 如果字体计算失败，使用估算值
                text_width = len(label_text) * 12
                text_height = 20
        else:
            # 没有字体时的估算
            text_width = len(label_text) * 12
            text_height = 20

        # 文本背景位置 - 确保在图像范围内
        text_bg_y = max(y_new - text_height - 8, 5)  # 增加上边距
        text_bg_x1 = max(x_new, 5)
        text_bg_x2 = min(x_new + text_width + 15, img_width - 5)

        # 绘制文本背景
        draw.rectangle([text_bg_x1, text_bg_y, text_bg_x2, text_bg_y + text_height + 5],
                       fill=color)

        # 文本位置
        text_x = text_bg_x1 + 5
        text_y = text_bg_y + 3

        # 绘制文本 - 使用更健壮的方法
        try:
            if font:
                draw.text((text_x, text_y), label_text, fill="white", font=font)
            else:
                # 如果没有字体，尝试使用默认字体
                try:
                    default_font = ImageFont.load_default()
                    draw.text((text_x, text_y), label_text, fill="white", font=default_font)
                except:
                    # 最后尝试不使用字体
                    draw.text((text_x, text_y), label_text, fill="white")
        except Exception as e:
            print(f"文本渲染失败: {e}")
            # 如果所有方法都失败，至少绘制一个简单的文本
            try:
                draw.text((int(x_new), int(y_new)), "X", fill="white")
            except:
                pass

    return image


def detect_stamen(pil_img: Image.Image, threshold: float = 0.5):
    """检测雄蕊 - 修复数据解析问题"""
    # 确保图像是RGB模式
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    # 调整图像尺寸以优化传输
    original_size = pil_img.size
    pil_img = resize_image_for_detection(pil_img)
    print(f"图像尺寸: {original_size} -> {pil_img.size}")

    # 准备图像数据
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=75)
    image_data = base64.b64encode(buf.getvalue()).decode('ascii')

    # 验证数据完整性
    data_length = len(image_data)
    print(f"Base64数据长度: {data_length} 字符")

    # 构建请求数据
    payload = {"image": image_data}
    payload_json = json.dumps(payload)
    content_length = len(payload_json)

    headers = {
        "Authorization": f"token {TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(content_length),
        "Connection": "close"
    }

    print(f"请求数据长度 (JSON): {content_length} 字符")

    # 重试机制
    for attempt in range(MAX_RETRIES):
        try:
            print(f"发送API请求... (尝试 {attempt + 1}/{MAX_RETRIES})")

            # 发送请求
            resp = requests.post(API_URL, data=payload_json, headers=headers, timeout=60)

            # 检查响应状态
            if resp.status_code != 200:
                error_msg = f"API请求失败: {resp.status_code}"
                if resp.text:
                    error_msg += f" - {resp.text[:100]}"
                raise RuntimeError(error_msg)

            # 解析响应
            response_data = resp.json()
            print("API响应成功")

            # 创建检测结果对象
            detection_result = DetectionResult(response_data, pil_img.size, threshold)

            # 在原图上绘制检测框
            boxed_img = pil_img.copy()
            if detection_result.detected_objects:
                boxed_img = draw_detection_boxes(boxed_img, detection_result.detected_objects)

            return boxed_img, detection_result

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ConnectionResetError) as e:
            print(f"网络连接错误 (尝试 {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                raise RuntimeError("网络连接不稳定，请检查网络连接后重试")

        except requests.exceptions.RequestException as e:
            print(f"网络请求错误 (尝试 {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                raise RuntimeError(f"网络请求错误: {str(e)}")

        except Exception as e:
            print(f"检测过程中出现错误 (尝试 {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                raise RuntimeError(f"检测失败: {str(e)}")

    raise RuntimeError("检测失败，请稍后重试")


# 添加亮度调节和直方图功能
import matplotlib

matplotlib.use('Agg')  # 避免GUI问题
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageEnhance


def adjust_image_brightness(pil_img: Image.Image, brightness_factor: float):
    """调整图像亮度"""
    if brightness_factor == 1.0:
        return pil_img

    enhancer = ImageEnhance.Brightness(pil_img)
    adjusted_img = enhancer.enhance(brightness_factor)
    return adjusted_img


def generate_brightness_histogram(pil_img: Image.Image):
    """生成亮度直方图"""
    try:
        # 转换为灰度图
        if pil_img.mode != 'L':
            gray_img = pil_img.convert('L')
        else:
            gray_img = pil_img

        # 获取像素数据
        pixels = np.array(gray_img).flatten()

        # 创建直方图
        plt.figure(figsize=(8, 3))
        plt.hist(pixels, bins=256, range=(0, 255), alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('亮度值')
        plt.ylabel('像素数量')
        plt.title('图像亮度分布')
        plt.grid(True, alpha=0.3)

        # 计算统计信息
        mean_brightness = np.mean(pixels)
        std_brightness = np.std(pixels)

        # 添加统计信息
        stats_text = f'平均亮度: {mean_brightness:.1f}\n标准差: {std_brightness:.1f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # 保存图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        histogram_img = Image.open(buf)
        plt.close()

        return histogram_img

    except Exception as e:
        print(f"生成亮度直方图失败: {e}")
        # 返回一个简单的错误图像
        error_img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(error_img)
        draw.text((50, 80), "亮度直方图生成失败", fill='red')
        return error_img


def detect_stamen_with_brightness(pil_img: Image.Image, threshold: float = 0.5, brightness_factor: float = 1.0):
    """带亮度调节的雄蕊检测"""
    # 调整亮度
    if brightness_factor != 1.0:
        pil_img = adjust_image_brightness(pil_img, brightness_factor)

    # 生成亮度直方图
    histogram_img = generate_brightness_histogram(pil_img)

    # 执行原有的检测逻辑
    boxed_img, detection_result = detect_stamen(pil_img, threshold)

    return boxed_img, detection_result, histogram_img
