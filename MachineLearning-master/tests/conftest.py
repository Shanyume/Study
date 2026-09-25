# -*- coding: utf-8 -*-
"""
pytest 配置：把项目根目录加入 sys.path，方便导入各模块。
"""
import sys
from pathlib import Path

# 项目根目录（tests/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
