import whisper
import numpy as np
import sounddevice as sd
import queue
import threading
import sys

# 配置参数
SAMPLE_RATE = 16000  # Whisper 使用 16kHz 采样率
CHUNK_DURATION = 5   # 每 5 秒处理一次音频
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# 加载 Whisper 模型
print("正在加载 Whisper 模型...")
model = whisper.load_model("large")
print("模型加载完成！")

# 音频队列
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """音频输入回调函数"""
    if status:
        print(f"音频状态: {status}", file=sys.stderr)
    # 将音频数据放入队列
    audio_queue.put(indata.copy())

def transcribe_audio(audio_data):


    """转录音频数据 - 使用高级 API，自动处理任意长度"""
    try:
        # 将音频数据转换为 float32 并展平
        audio = audio_data.flatten().astype(np.float32)
        
        # 使用 model.transcribe() 方法
        # 优点：
        # 1. 自动处理任意长度的音频（不需要手动 pad/trim）
        # 2. 对于长音频会自动分段处理
        # 3. 返回完整的转录结果，包括时间戳等信息
        result = model.transcribe(
            audio,
            language=None,  # 自动检测语言
            fp16=True,      # 使用 FP16 加速（如果有 GPU）
            verbose=False   # 不显示详细日志
        )
        
        return result['text'], result['language']
        
    except Exception as e:
        print(f"转录错误: {e}")
        return None, None

def process_audio():
    """处理音频队列中的数据"""
    audio_buffer = []
    
    while True:
        try:
            # 从队列获取音频数据
            chunk = audio_queue.get()
            audio_buffer.append(chunk)
            
            # 当缓冲区累积到指定时长时进行转录
            if len(audio_buffer) * len(audio_buffer[0]) >= CHUNK_SIZE:
                # 合并音频数据
                audio_data = np.concatenate(audio_buffer)
                
                # 转录
                text, lang = transcribe_audio(audio_data)
                
                if text and text.strip():
                    print(f"\n[{lang}] {text}")
                    print("-" * 50)
                
                # 清空缓冲区
                audio_buffer = []
                
        except KeyboardInterrupt:
            break

def main():
    """主函数"""
    print(f"\n开始实时语音转录 (优化版本)...")
    print(f"采样率: {SAMPLE_RATE} Hz")
    print(f"处理间隔: {CHUNK_DURATION} 秒")
    print(f"使用高级 API - 自动处理任意长度音频")
    print(f"按 Ctrl+C 停止\n")
    print("=" * 50)
    
    # 启动音频处理线程
    processing_thread = threading.Thread(target=process_audio, daemon=True)
    processing_thread.start()
    
    try:
        # 开始录音
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,  # 单声道
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.5)  # 每 0.5 秒回调一次
        ):
            print("🎤 正在监听麦克风...")
            # 保持运行
            processing_thread.join()
    except KeyboardInterrupt:
        print("\n\n停止录音...")
    except Exception as e:
        print(f"\n错误: {e}")

if __name__ == "__main__":
    main()

