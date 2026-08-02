"""
FSTN-4D V3 会话自动加载器
=========================
V1 auto_loader 的 V3 版本。Hermes Agent 每轮对话前调用 analyze()，
获得情绪指导并融入思考过程。

用法（与 V1 相同）：
    import sys; sys.path.insert(0, r'C:\\Users\\33196\\Desktop\\fstn_enhancement\\v3')
    from auto_loader_v3 import analyze
    guidance = analyze(USER_MESSAGE)
"""

import sys
import os

_V3_DIR = os.path.dirname(os.path.abspath(__file__))
if _V3_DIR not in sys.path:
    sys.path.insert(0, _V3_DIR)
from hermes_adapter_v3 import HermesFSTNAdapterV3

# 单例适配器（复用状态，跨轮保持情绪/记忆/学习）
_adapter: HermesFSTNAdapterV3 = None


def get_adapter() -> HermesFSTNAdapterV3:
    global _adapter
    if _adapter is None:
        _adapter = HermesFSTNAdapterV3()
    return _adapter


def analyze(user_message: str, context: str = "") -> str:
    """
    分析用户消息，返回可嵌入思考过程的指导文本。
    同时触发：情绪检测、感知更新、记忆存储、耦合学习。
    """
    adapter = get_adapter()
    result = adapter.analyze(user_message, context)
    return result["guidance_text"]


def analyze_full(user_message: str, context: str = "") -> dict:
    """返回完整分析结果（调试用）"""
    return get_adapter().analyze(user_message, context)


def save():
    """持久化引擎状态"""
    get_adapter().save()


if __name__ == "__main__":
    print(analyze("今天工作被批评了，好难过。"))
    save()
    print("✅ V3 auto_loader 工作正常")
