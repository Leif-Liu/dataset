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
model = whisper.load_model("turbo")
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
    """转录音频数据"""
    try:
        # 将音频数据转换为 float32 并展平
        audio = audio_data.flatten().astype(np.float32)
        
        # 方法1: 使用底层 API（当前方法）
        # 填充或裁剪到 30 秒（Whisper 的标准长度）
        # 如果音频 < 30秒，会在末尾用零填充；如果 > 30秒，会裁剪
        audio = whisper.pad_or_trim(audio)
        
        # 生成 log-Mel 频谱图
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        
        # 检测语言
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        
        # 解码音频
        options = whisper.DecodingOptions(language=detected_lang)
        result = whisper.decode(model, mel, options)
        
        return result.text, detected_lang
        
        # 方法2: 使用高级 API（更推荐，自动处理任意长度）
        # result = model.transcribe(audio, language=None)
        # return result['text'], result['language']
        
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
                
                # 清空缓冲区（可以保留一些重叠以提高准确性）
                audio_buffer = []
                
        except KeyboardInterrupt:
            break

def main():
    """主函数"""
    print(f"\n开始实时语音转录...")
    print(f"采样率: {SAMPLE_RATE} Hz")
    print(f"处理间隔: {CHUNK_DURATION} 秒")
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

