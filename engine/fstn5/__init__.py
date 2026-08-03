# -*- coding: utf-8 -*-
"""
fstn5/__init__.py — FSTN-5D 自进化内核

领域无关的经验驱动自改进引擎。独立部署（不依赖 Hermes）。
"""

from .core import FSTN5Core
from .models import (
    Experience, Strategy, UserVector,
    STRATEGY_ACTIVE, STRATEGY_FROZEN, STRATEGY_DEPRECATED,
)

__all__ = [
    "FSTN5Core",
    "Experience", "Strategy", "UserVector",
    "STRATEGY_ACTIVE", "STRATEGY_FROZEN", "STRATEGY_DEPRECATED",
]

__version__ = "5.0.0"
