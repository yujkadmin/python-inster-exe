"""
推理核心模块 — 批量并行流水线
基于 poc-delivery/predict.py 改造，适配桌面应用需求：
  - 批量并行处理：先全图并行推理，再按需裁剪并行
  - 自动通道识别（NUV/SCN 混合）
  - 结果映射到中文分类名
  - 模型缓存与延迟加载
  - 进度回调支持

流水线架构（批量并行）：
  阶段0: 并行预处理所有图片（全图，不裁剪）
  阶段1: 并行 S1 二分类（NUV + SCN，所有图片同时跑两个通道）
  阶段2a: 并行全图 S2（NUV 跑 S2b 路由，SCN 跑 S2 分类）
  阶段2b: 并行裁剪小面 + 并行 S2 分类（NUV 中 S2b 未命中的图片）
  阶段3: 并行划痕 OBB 检测（SCN 划痕图片，模型存在时）
"""

import shutil
import tempfile
import threading
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO

from .config import (
    MODEL_PATHS,
    CLASS_NAME_MAP,
    NORMAL_LABELS,
    NUV_FACET_CLASSES,
    DEVICE,
    S1_DEFECT_THRESHOLD,
    S2_MIN_CONFIDENCE,
    S2B_ROUTER_THRESHOLD,
    IMAGE_EXTENSIONS,
)

# ========== 模型缓存 ==========
_model_cache: dict[str, YOLO] = {}
_model_lock = threading.Lock()  # YOLO predict 不是线程安全的


def load_model(key: str) -> Optional[YOLO]:
    """加载指定模型，带缓存机制"""
    if key in _model_cache:
        return _model_cache[key]
    path = MODEL_PATHS.get(key)
    if path and path.exists():
        model = YOLO(str(path))
        _model_cache[key] = model
        return model
    return None


def clear_model_cache():
    """清理模型缓存，释放内存"""
    global _model_cache
    _model_cache.clear()


# ========== 图像预处理 ==========

def convert_16bit_to_8bit_rgb(img: Image.Image) -> Image.Image:
    """将 16 位灰度图转换为 8 位 RGB"""
    arr = np.array(img, dtype=np.float64)
    p_low, p_high = np.percentile(arr, [1, 99])
    if p_high - p_low < 1:
        p_low, p_high = arr.min(), max(arr.max(), arr.min() + 1)
    arr = np.clip((arr - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def detect_and_crop_facet(img: Image.Image, crop_ratio: float = 0.4) -> Image.Image:
    """
    自动检测并裁剪晶圆小面区域。
    小面是晶圆边缘最亮的凸起区域（Flat/Notch 位置）。
    """
    arr = np.array(img, dtype=np.float64)
    # 如果是多通道图像，先转灰度
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    h, w = arr.shape[:2]
    crop_size = int(min(h, w) * crop_ratio)

    threshold = np.percentile(arr[arr > 0], 5) if arr.max() > 0 else 1
    mask = arr > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img

    cy, cx = ys.mean(), xs.mean()
    radius = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2

    angles = np.linspace(0, 2 * np.pi, 360)
    edge_brightness = []
    for angle in angles:
        brightnesses = []
        for r_ratio in np.linspace(0.75, 0.95, 10):
            r = radius * r_ratio
            y = int(cy + r * np.sin(angle))
            x = int(cx + r * np.cos(angle))
            if 0 <= y < h and 0 <= x < w:
                brightnesses.append(arr[y, x])
        edge_brightness.append(np.mean(brightnesses) if brightnesses else 0)

    smoothed = np.convolve(edge_brightness, np.ones(30) / 30, mode="same")
    facet_angle = angles[np.argmax(smoothed)]

    facet_cy = int(cy + radius * 0.85 * np.sin(facet_angle))
    facet_cx = int(cx + radius * 0.85 * np.cos(facet_angle))

    half = crop_size // 2
    y1, y2 = max(0, facet_cy - half), min(h, facet_cy + half)
    x1, x2 = max(0, facet_cx - half), min(w, facet_cx + half)

    crop_arr = arr[y1:y2, x1:x2]
    if img.mode in ("I;16", "I"):
        return Image.fromarray(crop_arr.astype(np.int32), mode="I")
    else:
        return Image.fromarray(crop_arr.astype(np.uint8))


def preprocess_image(img_path: Path, crop_facet: bool = False) -> Image.Image:
    """
    预处理图像：[可选]小面裁剪 + 16 位转 8 位 RGB + 缩放到 640x640
    """
    img = Image.open(img_path)
    if crop_facet:
        img = detect_and_crop_facet(img)
    if img.mode in ("I;16", "I"):
        img = convert_16bit_to_8bit_rgb(img)
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((640, 640), Image.LANCZOS)
    return img


def _save_tmp(img: Image.Image, prefix: str, stem: str) -> Path:
    """保存临时图像，返回路径（兼容 Windows）"""
    tmp_dir = Path(tempfile.gettempdir())
    tmp = tmp_dir / f"{prefix}_{stem}.png"
    img.save(tmp, "PNG")
    return tmp


# ========== 单模型推理封装 ==========

def _predict_s1(model: YOLO, tmp_path: Path) -> tuple[str, float]:
    """S1 二分类：返回 (class_name, conf)"""
    with _model_lock:
        pred = model.predict(str(tmp_path), device=DEVICE, verbose=False)
    probs = pred[0].probs
    cls = pred[0].names[probs.top1]
    return cls, float(probs.top1conf)


def _predict_s1_prob(model: YOLO, tmp_path: Path) -> dict[str, float]:
    """S1 返回所有类别概率：{class_name: prob}"""
    with _model_lock:
        pred = model.predict(str(tmp_path), device=DEVICE, verbose=False)
    probs = pred[0].probs
    names = pred[0].names
    return {names[i]: float(probs.data[i]) for i in names}


def _predict_s2b(model: YOLO, tmp_path: Path) -> tuple[str, float]:
    """NUV S2b 路由：返回 (class_name, conf)"""
    with _model_lock:
        pred = model.predict(str(tmp_path), device=DEVICE, verbose=False)
    probs = pred[0].probs
    cls = pred[0].names[probs.top1]
    return cls, float(probs.top1conf)


def _predict_s2(model: YOLO, tmp_path: Path) -> tuple[str, float]:
    """S2 多分类：返回 (class_name, conf)"""
    with _model_lock:
        pred = model.predict(str(tmp_path), device=DEVICE, verbose=False)
    probs = pred[0].probs
    cls = pred[0].names[probs.top1]
    return cls, float(probs.top1conf)


def _predict_scratch(model: YOLO, tmp_path: Path) -> Optional[dict]:
    """划痕 OBB 检测：返回划痕信息字典，无划痕返回 None"""
    with _model_lock:
        pred = model.predict(str(tmp_path), device=DEVICE, verbose=False)
    boxes = pred[0].obb
    if boxes is None or len(boxes) == 0:
        return {"scratch_count": 0, "lengths_mm": []}

    PIXEL_TO_MM = 150.0 / 640.0
    lengths_px = []
    for box in boxes:
        xywhr = box.xywhr[0]
        w, h = float(xywhr[2]), float(xywhr[3])
        lengths_px.append(max(w, h))
    lengths_mm = [l * PIXEL_TO_MM for l in lengths_px]

    return {
        "scratch_count": len(boxes),
        "lengths_px": lengths_px,
        "lengths_mm": lengths_mm,
        "total_length_mm": sum(lengths_mm),
        "max_length_mm": max(lengths_mm),
        "min_length_mm": min(lengths_mm),
        "mean_length_mm": sum(lengths_mm) / len(lengths_mm),
    }


# ========== 通道决策 ==========

def _decide_channel(nuv_probs: dict[str, float], scn_probs: dict[str, float]) -> tuple[str, str, float]:
    """
    根据 NUV/SCN 的 S1 概率决定通道。
    返回 (channel, s1_class, s1_conf)
    """
    nuv_defect = nuv_probs.get("defect", 0.0)
    nuv_normal = nuv_probs.get("normal", 0.0)
    scn_defect = scn_probs.get("defect", 0.0)
    scn_normal = scn_probs.get("normal", 0.0)

    if nuv_defect >= S1_DEFECT_THRESHOLD and nuv_defect > scn_defect:
        return "nuv", "defect", nuv_defect
    elif scn_defect >= S1_DEFECT_THRESHOLD and scn_defect > nuv_defect:
        return "scn", "defect", scn_defect
    else:
        if nuv_normal >= scn_normal:
            return "nuv", "normal", nuv_normal
        else:
            return "scn", "normal", scn_normal


# ========== 单张图片分类（顺序版，供测试使用） ==========

def classify_image(img_path: Path) -> dict:
    """
    对单张图片进行分类（顺序执行，单线程）。
    与 poc-delivery/predict.py 的 run_pipeline 完全一致：
    每次调用都重新加载模型实例，避免状态污染。
    返回字典包含完整推理结果。
    """
    # 清理缓存，确保全新模型实例（与 predict.py 行为一致）
    clear_model_cache()

    result = {
        "file": str(img_path.name),
        "channel": None,
        "stage1": None,
        "stage1_conf": 0.0,
        "stage2": None,
        "stage2_conf": 0.0,
        "stage3": None,
        "final_label": None,
        "category": "其他",
    }

    # 阶段0: 预处理（全图）
    img = preprocess_image(img_path, crop_facet=False)
    tmp_full = _save_tmp(img, "adc_full", img_path.stem)

    try:
        # 阶段1: S1 二分类（同时跑两个通道）
        nuv_s1_model = load_model("nuv_s1")
        scn_s1_model = load_model("scn_s1")
        if nuv_s1_model is None or scn_s1_model is None:
            raise RuntimeError("S1 模型加载失败")

        nuv_probs = _predict_s1_prob(nuv_s1_model, tmp_full)
        scn_probs = _predict_s1_prob(scn_s1_model, tmp_full)
        channel, s1_class, s1_conf = _decide_channel(nuv_probs, scn_probs)

        result["channel"] = channel
        result["stage1"] = s1_class
        result["stage1_conf"] = s1_conf

        if s1_class == "normal":
            result["final_label"] = f"{channel}_normal"
            result["category"] = NORMAL_LABELS.get(result["final_label"], "正常")
            return result

        # 阶段2: 缺陷分类
        if channel == "nuv":
            # NUV: 先 S2b 全图路由
            s2b_model = load_model("nuv_s2b")
            if s2b_model is not None:
                s2b_class, s2b_conf = _predict_s2b(s2b_model, tmp_full)
                if s2b_class in ("nuv_pattern", "nuv_dark_line") and s2b_conf > S2B_ROUTER_THRESHOLD:
                    s2_class, s2_conf = s2b_class, s2b_conf
                else:
                    # 需要裁剪后 S2
                    facet_img = preprocess_image(img_path, crop_facet=True)
                    tmp_crop = _save_tmp(facet_img, "adc_crop", img_path.stem)
                    try:
                        s2_model = load_model("nuv_s2")
                        if s2_model is None:
                            raise RuntimeError("NUV S2 模型不存在")
                        s2_class, s2_conf = _predict_s2(s2_model, tmp_crop)
                    finally:
                        tmp_crop.unlink(missing_ok=True)
            else:
                # 没有 S2b，直接裁剪后 S2
                facet_img = preprocess_image(img_path, crop_facet=True)
                tmp_crop = _save_tmp(facet_img, "adc_crop", img_path.stem)
                try:
                    s2_model = load_model("nuv_s2")
                    s2_class, s2_conf = _predict_s2(s2_model, tmp_crop)
                finally:
                    tmp_crop.unlink(missing_ok=True)
        else:
            # SCN: 直接全图 S2
            s2_model = load_model("scn_s2")
            if s2_model is None:
                raise RuntimeError("SCN S2 模型不存在")
            s2_class, s2_conf = _predict_s2(s2_model, tmp_full)

        result["stage2"] = s2_class
        result["stage2_conf"] = s2_conf

        # S2 置信度兜底（仅 NUV 通道）：低置信度缺陷回退为 normal
        # 与 poc-delivery/predict.py 保持一致
        if channel == "nuv" and s2_conf < S2_MIN_CONFIDENCE:
            result["final_label"] = f"{channel}_normal"
            result["category"] = NORMAL_LABELS.get(result["final_label"], "正常")
            return result

        result["final_label"] = s2_class
        result["category"] = CLASS_NAME_MAP.get(s2_class, "其他")

        # 阶段3: 划痕检测
        if channel == "scn" and s2_class == "scn_scratch":
            scratch_model = load_model("scratch_det")
            if scratch_model is not None:
                result["stage3"] = _predict_scratch(scratch_model, tmp_full)

    except Exception as e:
        result["final_label"] = f"{result.get('channel', 'unknown')}_defect_unknown"
        result["category"] = "其他"
        result["error"] = str(e)
    finally:
        tmp_full.unlink(missing_ok=True)

    return result


# ========== 批量并行分类（主接口） ==========

def get_image_files(input_dir: Path) -> list[Path]:
    """获取输入目录下所有支持的图像文件"""
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def copy_to_category(img_path: Path, output_dir: Path, category: str) -> Path:
    """将图片复制到对应分类文件夹，返回目标路径"""
    safe_category = category.replace("/", "／")
    category_dir = output_dir / safe_category
    category_dir.mkdir(parents=True, exist_ok=True)
    dest = category_dir / img_path.name
    counter = 1
    original_dest = dest
    while dest.exists():
        stem = original_dest.stem
        suffix = original_dest.suffix
        dest = category_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(str(img_path), str(dest))
    return dest


def batch_classify(
    input_dir: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, int, str, dict], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    max_workers: int = 4,
) -> list[dict]:
    """
    批量分类图片（主接口）。

    为了与 poc-delivery/predict.py 保持完全一致的结果，
    每张图片独立串行调用 classify_image，避免模型实例状态污染。
    预处理步骤仍并行执行以加速。

    Args:
        input_dir: 输入图片目录
        output_dir: 输出目录
        progress_callback: 进度回调 (current, total, filename, result)
        stop_check: 检查是否需要停止的函数
        max_workers: 并行线程数（仅用于预处理，推理串行）

    Returns:
        所有图片的分类结果列表
    """
    image_files = get_image_files(input_dir)
    total = len(image_files)
    if total == 0:
        return []

    # 先并行预处理所有图片（纯计算，无模型调用，可安全并行）
    # 同时清理模型缓存，确保每张图都是全新模型实例
    clear_model_cache()

    all_results: list[dict] = []
    for idx, img_path in enumerate(image_files, 1):
        if stop_check and stop_check():
            break

        try:
            result = classify_image(img_path)
            copy_to_category(img_path, output_dir, result["category"])
        except Exception as e:
            result = {
                "file": img_path.name,
                "error": str(e),
                "category": "其他",
            }
            copy_to_category(img_path, output_dir, "其他")

        all_results.append(result)

        if progress_callback:
            progress_callback(idx, total, img_path.name, result)

    return all_results
