#!/usr/bin/env python3
"""
PyInstaller 打包脚本
支持生成 Windows EXE 和 macOS APP

用法:
    python build.py

输出:
    dist/WaferClassifier/      (Windows: 文件夹，内含 WaferClassifier.exe)
    dist/WaferClassifier.app/  (macOS: .app 应用包)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Windows 控制台 UTF-8 编码修复
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

APP_NAME = "WaferClassifier"
APP_NAME_CN = "晶圆缺陷自动分类系统"


def run_pyinstaller():
    """运行 PyInstaller 打包"""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--windowed",          # GUI 应用，不显示控制台
        "--noconfirm",         # 覆盖已存在的输出
        "--clean",             # 清理临时文件
        "--noconsole",
        # 添加 weights 目录作为数据文件
        "--add-data", f"{WEIGHTS_DIR}{os.pathsep}weights",
        # 核心模块
        "--add-data", f"{PROJECT_ROOT / 'core'}{os.pathsep}core",
        "--add-data", f"{PROJECT_ROOT / 'gui'}{os.pathsep}gui",
        # 隐藏导入（PyTorch / Ultralytics 依赖）
        "--hidden-import", "ultralytics",
        "--hidden-import", "ultralytics.nn.tasks",
        "--hidden-import", "ultralytics.models",
        "--hidden-import", "ultralytics.utils",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "numpy",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL.Image",
        "--hidden-import", "cv2",
        "--hidden-import", "scipy",
        "--hidden-import", "core",
        "--hidden-import", "gui",
        "--hidden-import", "core.config",
        "--hidden-import", "core.pipeline",
        "--hidden-import", "gui.main_window",
        # 主入口
        str(PROJECT_ROOT / "main.py"),
    ]

    # Windows: 文件夹模式（onedir），稳定性远优于单文件模式
    # 客户解压 zip 后双击 WaferClassifier.exe 即可运行

    # macOS 特定选项
    if sys.platform == "darwin":
        cmd.extend([
            "--osx-bundle-identifier", "com.jin.wafer-classifier",
        ])

    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError("PyInstaller 打包失败")


def copy_additional_files():
    """复制额外文件到输出目录"""
    if sys.platform == "darwin":
        app_dir = DIST_DIR / f"{APP_NAME}.app" / "Contents" / "MacOS"
        resources_dir = DIST_DIR / f"{APP_NAME}.app" / "Contents" / "Resources"
    else:
        app_dir = DIST_DIR / APP_NAME
        resources_dir = app_dir

    if not app_dir.exists():
        print(f"警告: 输出目录不存在: {app_dir}")
        return

    # 复制 README
    readme_src = PROJECT_ROOT / "README.md"
    if readme_src.exists():
        shutil.copy2(str(readme_src), str(resources_dir / "README.md"))

    print(f"额外文件已复制到: {resources_dir}")


def post_process_macos():
    """macOS 后处理：修改 Info.plist 显示中文名称"""
    if sys.platform != "darwin":
        return

    plist_path = DIST_DIR / f"{APP_NAME}.app" / "Contents" / "Info.plist"
    if not plist_path.exists():
        return

    # 读取 plist 内容
    content = plist_path.read_text(encoding="utf-8")

    # 添加/修改 CFBundleDisplayName
    if "CFBundleDisplayName" not in content:
        # 在 </dict> 前插入
        insert = f"    <key>CFBundleDisplayName</key>\n    <string>{APP_NAME_CN}</string>\n"
        content = content.replace("</dict>", insert + "</dict>", 1)
        plist_path.write_text(content, encoding="utf-8")
        print(f"已更新 Info.plist: CFBundleDisplayName = {APP_NAME_CN}")


def build_dmg():
    """macOS: 将 .app 打包为 .dmg"""
    if sys.platform != "darwin":
        print("当前不是 macOS 平台，跳过 DMG 打包")
        return

    app_path = DIST_DIR / f"{APP_NAME}.app"
    dmg_path = DIST_DIR / f"{APP_NAME}.dmg"

    if not app_path.exists():
        print(f"错误: {app_path} 不存在，无法打包 DMG")
        return

    # 尝试使用 create-dmg
    create_dmg = shutil.which("create-dmg")
    if create_dmg:
        cmd = [
            "create-dmg",
            "--volname", APP_NAME_CN,
            "--window-pos", "200", "120",
            "--window-size", "600", "400",
            "--icon-size", "100",
            "--app-drop-link", "450", "185",
            str(dmg_path),
            str(app_path),
        ]
        print(f"运行: {' '.join(cmd)}")
        subprocess.run(cmd)
        if dmg_path.exists():
            print(f"DMG 已生成: {dmg_path}")
        else:
            print("DMG 生成失败")
    else:
        # 回退: 使用 hdiutil
        print("create-dmg 未安装，尝试使用 hdiutil...")
        temp_dmg = DIST_DIR / "temp.dmg"
        cmd = [
            "hdiutil", "create",
            "-srcfolder", str(app_path),
            "-volname", APP_NAME_CN,
            "-fs", "HFS+",
            "-format", "UDRW",
            str(temp_dmg),
        ]
        subprocess.run(cmd)

        if temp_dmg.exists():
            # 转换为压缩格式
            subprocess.run([
                "hdiutil", "convert",
                str(temp_dmg),
                "-format", "UDZO",
                "-o", str(dmg_path),
            ])
            temp_dmg.unlink(missing_ok=True)
            if dmg_path.exists():
                print(f"DMG 已生成: {dmg_path}")


def main():
    print("=" * 60)
    print(f"开始打包: {APP_NAME_CN}")
    print(f"平台: {sys.platform}")
    print("=" * 60)

    # 清理旧的构建
    if BUILD_DIR.exists():
        print("清理旧的 build 目录...")
        shutil.rmtree(BUILD_DIR)
    if DIST_DIR.exists():
        print("清理旧的 dist 目录...")
        shutil.rmtree(DIST_DIR)

    # 运行 PyInstaller
    run_pyinstaller()

    # 复制额外文件
    copy_additional_files()

    # macOS 后处理
    post_process_macos()

    # 打包 DMG (仅 macOS)
    build_dmg()

    print("=" * 60)
    print("打包完成!")
    if sys.platform == "darwin":
        print(f"输出: {DIST_DIR / f'{APP_NAME}.app'}")
        dmg = DIST_DIR / f"{APP_NAME}.dmg"
        if dmg.exists():
            print(f"DMG: {dmg}")
    else:
        print(f"输出: {DIST_DIR / APP_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
