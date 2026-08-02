# -*- coding: utf-8 -*-
"""
打包 FSTN-4D v2 增强层为独立 ZIP。
"""
import os
import zipfile

BASE = r"C:\Users\33196\Desktop\fstn_enhancement"
OUT = r"C:\Users\33196\Desktop\fstn_4d_v2_enhancement.zip"

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
