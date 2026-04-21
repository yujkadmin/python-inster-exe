"""
配置模块：分类映射、模型路径、全局常量
"""
import os
import sys
from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录，兼容 PyInstaller 打包环境"""
    if getattr(sys, "frozen", False):
        meipass = Path(sys._MEIPASS)
        # PyInstaller macOS app bundle: --add-data 文件放在 Resources/ 下
        if sys.platform == "darwin" and meipass.name == "MacOS":
            resources = meipass.parent / "Resources"
            if (resources / "weights").exists():
                return resources
        return meipass
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# 模型文件路径配置
MODEL_PATHS = {
    "nuv_s1": WEIGHTS_DIR / "nuv_s1_binary.pt",
    "nuv_s2": WEIGHTS_DIR / "nuv_s2_defect.pt",
    "nuv_s2b": WEIGHTS_DIR / "nuv_s2b_nonfacet.pt",
    "scn_s1": WEIGHTS_DIR / "scn_s1_binary.pt",
    "scn_s2": WEIGHTS_DIR / "scn_s2_defect.pt",
}

# 支持的图像格式
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# 分类映射：模型内部标签 -> 用户中文分类文件夹名
CLASS_NAME_MAP = {
    "nuv_facet_dark": "NUV小面发黑",
    "nuv_white_line": "NUV/SCN-小面白线/黑线",
    "nuv_black_line": "NUV/SCN-小面白线/黑线",
    "scn_vertical_line": "ScN小面竖线",
    "nuv_irregular_shape": "NUV/SCN/SspfRO-小面不规则",
    "nuv_pattern": "NUV花纹花斑",
    "nuv_dark_line": "NUV黑线",
    "scn_scratch": "其他",
}

# 正常类标签映射
NORMAL_LABELS = {
    "nuv_normal": "正常",
    "scn_normal": "正常",
}

# 用户可见的 6 个主要分类（用于 UI 展示）
USER_CATEGORIES = [
    "NUV小面发黑",
    "NUV/SCN-小面白线/黑线",
    "ScN小面竖线",
    "NUV/SCN/SspfRO-小面不规则",
    "NUV花纹花斑",
    "NUV黑线",
]

# NUV 通道中需要裁剪小面区域的缺陷类别（S2 小面类）
NUV_FACET_CLASSES = {
    "nuv_facet_dark",
    "nuv_irregular_shape",
    "nuv_white_line",
    "nuv_black_line",
}

# 推理设备（强制 CPU，避免 MPS 兼容性问题）
DEVICE = "cpu"

# 置信度阈值（与 poc-delivery/predict.py 保持一致）
S1_DEFECT_THRESHOLD = 0.70  # NUV S1 defect 低于此值回退为 normal
S2_MIN_CONFIDENCE = 0.60    # NUV S2 低于此值回退为 normal
S2B_ROUTER_THRESHOLD = 0.75
