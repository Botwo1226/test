import wave
import json
import numpy as np
from vosk import Model, KaldiRecognizer
import pyttsx3
import tempfile
import os
import threading
from datetime import datetime
import queue
import time


class SpeechProcessor:
    def __init__(self, model_path="zh-3"):
        """初始化语音处理器 - 修复语音合成问题"""
        try:
            # 语音识别模型
            self.model = Model(model_path)
            print("✅ 语音识别模型加载成功")
        except Exception as e:
            print(f"❌ 语音识别模型加载失败: {e}")
            self.model = None

        # 语音合成引擎 - 使用单例模式
        self._tts_lock = threading.Lock()
        self._tts_queue = queue.Queue()
        self._tts_worker_running = True
        self._tts_worker_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_worker_thread.start()

        # 添加兼容性属性，避免应用代码报错
        self.tts_engine = True  # 表示TTS引擎可用
        self._tts_available = True

        print("✅ 语音合成引擎初始化成功")

    def _get_tts_engine(self):
        """获取新的TTS引擎实例 - 避免状态污染"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # 语速
            engine.setProperty('volume', 0.8)  # 音量
            return engine
        except Exception as e:
            print(f"❌ 创建TTS引擎失败: {e}")
            return None

    def _tts_worker(self):
        """TTS工作线程 - 处理语音合成请求"""
        while self._tts_worker_running:
            try:
                # 等待请求，最多等待1秒
                item = self._tts_queue.get(timeout=1)
                if item is None:  # 停止信号
                    break

                text, save_to_file, callback = item
                self._process_tts_request(text, save_to_file, callback)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ TTS工作线程错误: {e}")
                time.sleep(0.1)

    def _process_tts_request(self, text, save_to_file, callback):
        """处理单个TTS请求"""
        engine = None
        try:
            # 创建新的引擎实例，避免状态污染
            engine = self._get_tts_engine()
            if not engine:
                if callback:
                    callback(None)
                return

            if save_to_file:
                # 保存到临时文件
                temp_file = os.path.join(tempfile.gettempdir(),
                                         f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(text) % 10000}.mp3")
                engine.save_to_file(text, temp_file)
                engine.runAndWait()

                # 检查文件是否生成成功
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                    if callback:
                        callback(temp_file)
                else:
                    print("❌ 语音文件生成失败或为空")
                    if callback:
                        callback(None)
            else:
                # 直接播放
                engine.say(text)
                engine.runAndWait()
                if callback:
                    callback(None)

        except Exception as e:
            print(f"❌ 语音合成处理失败: {e}")
            if callback:
                callback(None)
        finally:
            # 清理引擎资源
            if engine:
                try:
                    engine.stop()
                    del engine
                except:
                    pass

    def speech_to_text(self, audio_file_path):
        """将语音转换为文本"""
        if not self.model:
            return "语音识别功能不可用"

        try:
            with wave.open(audio_file_path, "rb") as wf:
                print(f"音频参数 | 采样率: {wf.getframerate()}Hz, 声道: {wf.getnchannels()}, 位深: {wf.getsampwidth()}")

                # 创建识别器
                recognizer = KaldiRecognizer(self.model, wf.getframerate())

                print("开始语音识别...")
                full_text = ""

                while True:
                    raw_frames = wf.readframes(64000)
                    if not raw_frames:
                        break

                    # 音频数据处理（保持原有逻辑）
                    if wf.getsampwidth() == 3:
                        # 24位深度处理
                        data_u8 = np.frombuffer(raw_frames, dtype=np.uint8)
                        num_samples = len(data_u8) // 3
                        data = np.zeros(num_samples, dtype=np.int32)

                        for i in range(num_samples):
                            b0 = data_u8[i * 3]
                            b1 = data_u8[i * 3 + 1]
                            b2 = data_u8[i * 3 + 2]
                            raw_value = (b2 << 16) | (b1 << 8) | b0

                            if (raw_value & 0x800000):
                                data[i] = np.int32(raw_value | 0xFF000000)
                            else:
                                data[i] = np.int32(raw_value)

                        data = (data >> 8).astype(np.int16)
                    else:
                        # 其他位深度处理
                        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
                        data = np.frombuffer(raw_frames, dtype=dtype_map.get(wf.getsampwidth(), np.int16))
                        if wf.getsampwidth() != 2:
                            data = (data >> (8 * (wf.getsampwidth() - 2))).astype(np.int16)

                    # 立体声转单声道
                    if wf.getnchannels() == 2:
                        left = data[::2]
                        right = data[1::2]
                        data = (left + right) // 2

                    processed_frames = data.tobytes()

                    if recognizer.AcceptWaveform(processed_frames):
                        result = json.loads(recognizer.Result())
                        if 'text' in result and result['text']:
                            full_text += result['text'] + " "

                # 获取最终结果
                final_result = json.loads(recognizer.FinalResult())
                if 'text' in final_result and final_result['text']:
                    full_text += final_result['text']

                return full_text.strip() if full_text.strip() else "未识别到语音内容"

        except Exception as e:
            print(f"语音识别失败: {e}")
            return f"语音识别失败: {str(e)}"

    def text_to_speech(self, text, save_to_file=False):
        """将文本转换为语音 - 修复版本"""
        if not text or not text.strip():
            return None

        result = [None]  # 用于存储结果
        event = threading.Event()

        def callback(output):
            result[0] = output
            event.set()

        # 将请求加入队列
        self._tts_queue.put((text, save_to_file, callback))

        # 等待结果，设置超时
        if event.wait(timeout=30):  # 30秒超时
            return result[0]
        else:
            print("❌ 语音合成超时")
            return None

    def text_to_speech_async(self, text):
        """异步语音合成（不阻塞主线程）"""
        if not text or not text.strip():
            return

        def async_callback(output):
            if output:
                print(f"✅ 异步语音合成完成: {output}")
            else:
                print("❌ 异步语音合成失败")

        self._tts_queue.put((text, False, async_callback))

    def cleanup(self):
        """清理资源"""
        self._tts_worker_running = False
        self._tts_queue.put(None)  # 发送停止信号
        if self._tts_worker_thread.is_alive():
            self._tts_worker_thread.join(timeout=5)


# 创建全局实例
speech_processor = SpeechProcessor()
