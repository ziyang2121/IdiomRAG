#!/usr/bin/env python3
"""
下载 Hugging Face 模型到本地指定目录，用于离线部署。
默认下载 sentence-transformers/m3e-base。
"""

import os
import sys
import argparse
from pathlib import Path

def download_model(model_id: str, local_dir: str):
    """下载模型仓库到本地目录"""
    print(f"开始下载模型: {model_id}")
    print(f"目标目录: {local_dir}")

    # 确保 huggingface_hub 已安装
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("正在安装 huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    # 创建目标目录（如果不存在）
    local_path = Path(local_dir).expanduser().resolve()
    local_path.mkdir(parents=True, exist_ok=True)

    # 下载整个仓库（不建立符号链接，直接复制文件）
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,      # 复制文件，便于移动
        resume_download=True,              # 支持断点续传
        ignore_patterns=[                   # 可选：忽略非 PyTorch 权重文件
            "flax_model.msgpack",
            "tf_model.h5"
            			                 # 如果不需要 safetensors 可取消注释
        ]
    )
    print(f"✅ 模型下载完成！保存位置: {local_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 Hugging Face 模型到本地目录")
    parser.add_argument(
        "model_id",
        nargs="?",
        default="moka-ai/m3e-base",
        help="Hugging Face 模型ID (默认:moka-ai/m3e-base)"
    )
    parser.add_argument(
        "local_dir",
        nargs="?",
        default="./models/embedding/m3e-base",
        help="本地保存目录 (默认: ./models/embedding/m3e-base)"
    )
    args = parser.parse_args()

    download_model(args.model_id, args.local_dir)