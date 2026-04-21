# 晶圆缺陷自动分类系统 — 桌面应用

基于 YOLO 深度学习模型的 ADC（Automatic Defect Classification）智能检测工具，支持 NUV/SCN 双通道自动识别。

## 功能特性

- **批量导入图片**：支持 PNG、JPG、JPEG、BMP、TIFF、WebP 格式
- **自动通道识别**：自动判断每张图片属于 NUV 还是 SCN 通道
- **智能分类**：基于三级流水线（S1 二分类 + S2 多分类）精准识别缺陷类型
- **自动拆分输出**：按分类自动复制图片到对应文件夹
- **实时进度**：进度条 + 日志 + 分类统计表格
- **跨平台**：支持 Windows (.exe) 和 macOS (.app / .dmg)

## 输出分类

| 分类文件夹 | 说明 |
|-----------|------|
| NUV小面发黑 | 小面漆黑缺陷 |
| NUV/SCN-小面白线/黑线 | 小面白线、小面黑线 |
| ScN小面竖线 | SCN 通道小面竖线 |
| NUV/SCN/SspfRO-小面不规则 | 小面形状不规则 |
| NUV花纹花斑 | 花纹、花斑缺陷 |
| NUV黑线 | 通道图黑线 |
| 正常 | 无缺陷的正常晶圆 |
| 其他 | 划痕或其他未明确分类 |

## 开发环境运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
python main.py
```

## 打包为可执行文件

### Windows (生成 .exe)

```bash
python build.py
```

输出：`dist/WaferClassifier/` 文件夹（包含 `.exe` 及依赖）

### macOS (生成 .app + .dmg)

```bash
python build.py
```

输出：
- `dist/WaferClassifier.app` — macOS 应用包
- `dist/WaferClassifier.dmg` — DMG 安装包（如 create-dmg 或 hdiutil 可用）

## 技术栈

- **GUI**: PySide6 (Qt for Python)
- **深度学习**: Ultralytics YOLOv8 + PyTorch
- **图像处理**: Pillow, OpenCV, NumPy
- **打包**: PyInstaller

## 项目结构

```
wafer-adc-desktop/
├── main.py              # 应用入口
├── core/
│   ├── config.py        # 分类映射、模型路径配置
│   └── pipeline.py      # YOLO 推理核心
├── gui/
│   └── main_window.py   # PySide6 主界面
├── weights/             # 模型权重 (.pt)
│   ├── nuv_s1_binary.pt
│   ├── nuv_s2_defect.pt
│   ├── nuv_s2b_nonfacet.pt
│   ├── scn_s1_binary.pt
│   └── scn_s2_defect.pt
├── build.py             # PyInstaller 打包脚本
├── requirements.txt     # Python 依赖
└── README.md            # 本文件
```

## 已知限制

- 首次启动时模型加载需要几秒时间（模型文件总计约 40MB）
- 打包后体积较大（包含 PyTorch 运行库，约 500MB-1GB）
- 推理使用 CPU 模式，单张图片处理时间约 100-300ms
