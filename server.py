import os
import sys
import struct
import asyncio
import subprocess
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI(title="Qwen3-TTS HTTP Server")

# 默认情况下需要将 leaxer-qwen3-tts.exe 放到根目录或者指定路径
# 同样模型目录也需要在此配置
MODEL_DIR = os.getenv("TTS_MODEL_DIR", "onnx/onnx_kv_06b")
EXE_PATH = os.getenv("TTS_EXE_PATH", "./build/leaxer-qwen3-tts.exe")

# 如果在 Windows 环境运行，需要确保 exe 后缀
if os.name == 'nt' and not EXE_PATH.endswith('.exe'):
    if os.path.exists(EXE_PATH + '.exe'):
        EXE_PATH += '.exe'

class TTSDaemon:
    def __init__(self, exe_path, model_dir):
        self.exe_path = exe_path
        self.model_dir = model_dir
        self.process = None
        self.lock = asyncio.Lock()
        
    def start(self):
        if not os.path.exists(self.exe_path):
            print(f"WARNING: TTS Engine builtin executable not found at {self.exe_path}")
            print("Please compile the C++ project and ensure the path is correct.")
            return
            
        print(f"Starting TTS Daemon: {self.exe_path} -m {self.model_dir} --daemon")
        # 启动为子进程，并通过管道读写
        self.process = subprocess.Popen(
            [self.exe_path, "-m", self.model_dir, "--daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr, # 日志打印到控制台
            bufsize=0 # 必须采用无缓冲二进制模式
        )
        
    def stop(self):
        if self.process:
            try:
                self.process.stdin.write(b"EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except:
                self.process.kill()
            self.process = None

    async def synthesize(self, text: str, lang: str = "auto", seed: int = -1) -> bytes:
        if not self.process or self.process.poll() is not None:
            self.start() # 如果进程崩溃则尝试重启
            
        if not self.process:
            raise RuntimeError("TTS Daemon could not be started. Check EXE_PATH and MODEL_DIR.")

        async with self.lock:
            # 去除换行符以防止破坏通讯协议
            text_clean = text.replace('\n', '，').replace('\r', ' ')
            command = f"{lang}|||{seed}|||{text_clean}\n".encode('utf-8')
            
            # 发送生成命令
            self.process.stdin.write(command)
            self.process.stdin.flush()

            # 读取头部 "AUDIO <字节数>\n"
            header = b""
            while True:
                char = self.process.stdout.read(1)
                if not char or char == b'\n':
                    break
                header += char
                
            header_str = header.decode('utf-8').strip()
            if header_str.startswith("AUDIO "):
                size = int(header_str.split(" ")[1])
                # 读取二进制音频 PCM 数据
                audio_data = b""
                bytes_left = size
                while bytes_left > 0:
                    chunk = self.process.stdout.read(min(bytes_left, 4096))
                    if not chunk:
                        break
                    audio_data += chunk
                    bytes_left -= len(chunk)
                return audio_data
            elif header_str.startswith("ERROR"):
                raise RuntimeError("TTS Engine returned an error.")
            else:
                raise RuntimeError(f"Unexpected daemon response: {header_str}")

daemon = TTSDaemon(EXE_PATH, MODEL_DIR)

@app.on_event("startup")
async def startup():
    daemon.start()

@app.on_event("shutdown")
async def shutdown():
    daemon.stop()

def create_wav_header(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(pcm_data)
    file_size = 36 + data_size
    
    # 构造标准的 44 字节 WAV 文件头
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', file_size, b'WAVE',
        b'fmt ', 16, 1, channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
        b'data', data_size
    )
    return header

@app.get("/api/tts")
@app.post("/api/tts")
async def api_tts(text: str, lang: str = "auto", seed: int = -1):
    """
    供阅读APP调用的 HTTP TTS 接口（支持 GET 和 POST）
    返回的是标准的 WAV 音频流文件
    """
    try:
        pcm_data = await daemon.synthesize(text, lang, seed)
        wav_header = create_wav_header(pcm_data, sample_rate=24000)
        
        def iterfile():
            yield wav_header
            yield pcm_data
            
        return StreamingResponse(iterfile(), media_type="audio/wav")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("--------------------------------------------------")
    print(f"配置的模型路径: {MODEL_DIR}")
    print(f"配置的工具路径: {EXE_PATH}")
    print("注意: 首次运行请先使用 cmake 重新编译 C++ 源码产生支持 --daemon 的程序")
    print("启动服务器... 监听端口: 8000")
    print("--------------------------------------------------")
    uvicorn.run(app, host="0.0.0.0", port=8000)
