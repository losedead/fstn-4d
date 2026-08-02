# -*- coding: utf-8 -*-
"""
打包 FSTN-4D v2 增强层为独立 ZIP。
"""
import os
import zipfile

import os

# 相对于脚本所在目录，而非硬编码的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = SCRIPT_DIR  # fstn_enhancement 根目录
OUT = os.path.join(os.path.dirname(SCRIPT_DIR), "fstn_4d_v2_enhancement.zip")

FILES = [
    # 新增 v2 模块
    r"engine\v2_emotion_classifier.py",
    r"engine\v2_vector_retrieval.py",
    r"engine\v2_coupling_learner.py",
    r"engine\v2_engine.py",
    r"engine\v2_benchmark.py",
    r"engine\v2_retrieval_compare.py",
    # 依赖的 v1 模块（用于独立运行）
    r"engine\fstn_core.py",
    r"engine\fstn_emotion.py",
    r"engine\fstn_memory.py",
    r"engine\fstn_perception.py",
    r"engine\hermes_adapter.py",
    # 文档与基准
    r"v2_README.md",
    r"benchmark_report.json",
]

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in FILES:
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            zf.write(p, os.path.join("fstn_4d_v2", f))
            print(f"  + {f}")
        else:
            print(f"  ! 缺失: {f}")

print(f"\nZIP 已生成: {OUT}  ({os.path.getsize(OUT)} bytes)")
