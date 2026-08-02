# -*- coding: utf-8 -*-
"""
package_v4.py — 打包 FSTN-4D v4 Ultimate 补齐成果

产物：~/Desktop/fstn_4d_v4_ultimate.zip
包含：
  - engine/v4_*.py（HNSW/感知空间/通感图/引擎/基准/训练/评估）
  - engine/training_data/（400 样本 + 400 DPO 偏好 + 统计）
  - engine/v4_README.md
  - benchmark_report_v4.json（32 项基准结果）
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))  # 即 engine 目录
ENGINE = BASE
OUT = os.path.join(os.path.expanduser("~/Desktop"), "fstn_4d_v4_ultimate.zip")

# 打包清单（engine 目录下的 v4 相关 + 训练数据 + 文档）
INCLUDE_FILES = [
    "v4_hnsw_index.py",
    "v4_perceptual_space.py",
    "v4_engine.py",
    "v4_benchmark.py",
    "v4_README.md",
    "build_training_data.py",
    "dpo_reward.py",
    "finetune_fstn_lora.py",
    "finetune_fstn_dpo.py",
    "eval_fstn_model.py",
    "e2e_v4.py",
    "benchmark_report_v4.json",
]
INCLUDE_DIRS = [
    "training_data",
]


def run_benchmark():
    """打包前重跑基准，确保报告最新"""
    print("重跑 32 项基准...")
    r = subprocess.run([sys.executable, os.path.join(ENGINE, "v4_benchmark.py")],
                       capture_output=True, text=True, timeout=300)
    print("基准完成，通过:", "通过: 32/32" in r.stdout or "通过: 32/32" in r.stderr)
    return os.path.join(ENGINE, "benchmark_report_v4.json")


def main():
    report = run_benchmark()
    # 读基准报告摘要
    with open(report, encoding="utf-8") as f:
        data = json.load(f)

    tmp = tempfile.mkdtemp(prefix="hermes-fstn-pkg-")
    pkg_root = os.path.join(tmp, "fstn_4d_v4_ultimate")
    pkg_engine = os.path.join(pkg_root, "engine")
    os.makedirs(pkg_engine, exist_ok=True)

    # 复制文件
    for f in INCLUDE_FILES:
        src = os.path.join(ENGINE, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(pkg_engine, f))
    for d in INCLUDE_DIRS:
        src_dir = os.path.join(ENGINE, d)
        if os.path.isdir(src_dir):
            shutil.copytree(src_dir, os.path.join(pkg_engine, d),
                            ignore=shutil.ignore_patterns("__pycache__"))

    # 写结果摘要
    summary = {
        "engine": "FSTN4DEngineV4",
        "benchmark": {
            "passed": data.get("passed"),
            "total": data.get("total"),
            "rate": data.get("rate"),
        },
        "training_data": "training_data/train_fstn_4d.jsonl (400 条) + dpo_preferences.jsonl (400 对)",
        "note": "见 engine/v4_README.md 快速验证与训练说明",
    }
    with open(os.path.join(pkg_root, "SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打包
    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(pkg_root):
            for file in files:
                full = os.path.join(root, file)
                rel = os.path.relpath(full, tmp)
                zf.write(full, rel)

    shutil.rmtree(tmp)
    size = os.path.getsize(OUT) / 1024
    print(f"\n✅ 打包完成: {OUT}")
    print(f"   大小: {size:.1f} KB")
    print(f"   基准: {data.get('passed')}/{data.get('total')} 通过 ({data.get('rate')*100:.0f}%)")


if __name__ == "__main__":
    main()
