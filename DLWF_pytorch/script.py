import subprocess
import time
import os

# 你期望的最小可用显存（单位 MB）
MIN_FREE_MEM_MB = 8000  

def get_free_memory():
    """获取 GPU 剩余显存（MB）"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        text=True
    )
    mem_list = [int(x) for x in result.stdout.strip().split("\n")]
    return max(mem_list)  # 多卡时取最大可用的

while True:
    free_mem = get_free_memory()
    print(f"当前可用显存: {free_mem} MB")
    if free_mem >= MIN_FREE_MEM_MB:
        print("显存足够，启动任务...")
        os.system("python main_5000.py -m llm -t True")  # 启动你的任务
        break
    time.sleep(10)  # 每 10 秒检测一次