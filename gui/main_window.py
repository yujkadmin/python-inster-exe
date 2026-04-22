"""
PySide6 主窗口
"""

import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplashScreen,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QProgressBar,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QMessageBox,
    QHeaderView,
    QGroupBox,
    QSplitter,
)
from PySide6.QtCore import Qt, QThread, Signal, QMutex, QMutexLocker
from PySide6.QtGui import QFont, QIcon, QPixmap, QColor


# ========== 样式表 ==========
STYLE_SHEET = """
QMainWindow {
    background-color: #f5f6fa;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #dcdde1;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    padding-bottom: 10px;
    padding-left: 15px;
    padding-right: 15px;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #2f3640;
}
QPushButton {
    background-color: #487eb0;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 18px;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #40739e;
}
QPushButton:disabled {
    background-color: #dcdde1;
    color: #718093;
}
QPushButton#danger {
    background-color: #c23616;
}
QPushButton#danger:hover {
    background-color: #e84118;
}
QLineEdit {
    border: 1px solid #dcdde1;
    border-radius: 4px;
    padding: 6px 10px;
    background-color: #f5f6fa;
}
QProgressBar {
    border: 1px solid #dcdde1;
    border-radius: 4px;
    text-align: center;
    height: 20px;
}
QProgressBar::chunk {
    background-color: #44bd32;
    border-radius: 4px;
}
QTextEdit {
    border: 1px solid #dcdde1;
    border-radius: 4px;
    background-color: #2f3640;
    color: #f5f6fa;
    font-family: "SF Mono", "Consolas", "Courier New", monospace;
    font-size: 12px;
}
QTableWidget {
    border: 1px solid #dcdde1;
    border-radius: 4px;
    background-color: #ffffff;
    gridline-color: #f5f6fa;
}
QTableWidget::item {
    padding: 6px;
}
QHeaderView::section {
    background-color: #487eb0;
    color: white;
    padding: 6px;
    border: none;
    font-weight: bold;
}
QLabel#title {
    font-size: 20px;
    font-weight: bold;
    color: #2f3640;
}
QLabel#subtitle {
    font-size: 12px;
    color: #718093;
}
QLabel#status {
    font-size: 13px;
    color: #487eb0;
}
"""


# ========== 工作线程 ==========

class ClassifyWorker(QThread):
    progress = Signal(int, int, str, dict)  # current, total, filename, result
    finished_signal = Signal(list)  # results
    error = Signal(str)

    def __init__(self, input_dir: Path, output_dir: Path):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._stop_requested = False
        self._mutex = QMutex()

    def stop(self):
        with QMutexLocker(self._mutex):
            self._stop_requested = True

    def is_stopped(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._stop_requested

    def run(self):
        try:
            from core import batch_classify  # 延迟导入，避免启动慢
            # 根据 CPU 核心数自动设置线程数，最少 2 个，最多 8 个
            max_workers = max(2, min(os.cpu_count() or 4, 8))
            results = batch_classify(
                self.input_dir,
                self.output_dir,
                max_workers=max_workers,
                progress_callback=lambda c, t, f, r: self.progress.emit(c, t, f, r),
                stop_check=lambda: self.is_stopped(),
            )
            self.finished_signal.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ========== 主窗口 ==========

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("晶圆缺陷自动分类系统")
        self.setMinimumSize(960, 720)
        self.worker: Optional[ClassifyWorker] = None

        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # === 标题 ===
        title_layout = QVBoxLayout()
        title_label = QLabel("晶圆缺陷自动分类系统")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        subtitle_label = QLabel("基于 YOLO 深度学习模型的 ADC 智能检测工具")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        layout.addLayout(title_layout)

        # === 输入输出选择 ===
        io_group = QGroupBox("目录设置")
        io_layout = QVBoxLayout()

        # 输入目录
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入目录:"))
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("请选择包含晶圆图像的文件夹...")
        self.input_edit.setReadOnly(True)
        input_layout.addWidget(self.input_edit)
        self.input_btn = QPushButton("浏览...")
        self.input_btn.clicked.connect(self._choose_input)
        input_layout.addWidget(self.input_btn)
        self.input_count_label = QLabel("图像: 0 张")
        self.input_count_label.setStyleSheet("color: #718093; font-size: 12px;")
        input_layout.addWidget(self.input_count_label)
        io_layout.addLayout(input_layout)

        # 输出目录
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("请选择分类结果输出文件夹...")
        self.output_edit.setReadOnly(True)
        output_layout.addWidget(self.output_edit)
        self.output_btn = QPushButton("浏览...")
        self.output_btn.clicked.connect(self._choose_output)
        output_layout.addWidget(self.output_btn)
        io_layout.addLayout(output_layout)

        io_group.setLayout(io_layout)
        layout.addWidget(io_group)

        # === 操作按钮 ===
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setMinimumWidth(120)
        self.start_btn.clicked.connect(self._start_processing)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setMinimumWidth(120)
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # === 进度区域 ===
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("status")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        self.time_label = QLabel("")
        self.time_label.setStyleSheet("color: #718093; font-size: 12px;")
        status_layout.addWidget(self.time_label)
        progress_layout.addLayout(status_layout)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # === 日志与结果 ===
        splitter = QSplitter(Qt.Vertical)

        # 日志
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(150)
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)
        splitter.addWidget(log_group)

        # 结果统计
        result_group = QGroupBox("分类结果统计")
        result_layout = QVBoxLayout()
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["分类", "数量"])
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.result_table.setColumnWidth(1, 100)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        result_layout.addWidget(self.result_table)
        result_group.setLayout(result_layout)
        splitter.addWidget(result_group)

        splitter.setSizes([250, 200])
        layout.addWidget(splitter, stretch=1)

    def _apply_styles(self):
        self.setStyleSheet(STYLE_SHEET)

    def _choose_input(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入图片目录")
        if dir_path:
            path = Path(dir_path)
            self.input_edit.setText(str(path))
            from core import get_image_files  # 延迟导入
            files = get_image_files(path)
            self.input_count_label.setText(f"图像: {len(files)} 张")
            self._log(f"已选择输入目录: {path} (共 {len(files)} 张图像)")

    def _choose_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_edit.setText(dir_path)
            self._log(f"已选择输出目录: {dir_path}")

    def _log(self, message: str):
        self.log_edit.append(message)
        # 滚动到底部
        scrollbar = self.log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _start_processing(self):
        input_dir = Path(self.input_edit.text()) if self.input_edit.text() else None
        output_dir = Path(self.output_edit.text()) if self.output_edit.text() else None

        if not input_dir or not input_dir.exists():
            QMessageBox.warning(self, "提示", "请先选择有效的输入目录")
            return
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先选择输出目录")
            return

        from core import get_image_files  # 延迟导入
        image_files = get_image_files(input_dir)
        if not image_files:
            QMessageBox.warning(self, "提示", "输入目录中没有支持的图像文件")
            return

        self.status_label.setText("正在加载 AI 模型，请稍候...")
        self._log("正在加载 AI 模型...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.input_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_table.setRowCount(0)
        self.log_edit.clear()

        self._log(f"开始处理 {len(image_files)} 张图像...")
        self._log(f"输出目录: {output_dir}")
        self._log("-" * 50)

        self.worker = ClassifyWorker(input_dir, output_dir)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("正在停止...")
            self._log("用户请求停止处理...")

    def _on_progress(self, current: int, total: int, filename: str, result: dict):
        pct = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"正在处理 ({current}/{total}): {filename}")

        channel = result.get("channel", "?")
        category = result.get("category", "其他")
        s1 = result.get("stage1", "?")
        s1_conf = result.get("stage1_conf", 0.0)
        s2 = result.get("stage2", "-")
        s2_conf = result.get("stage2_conf", 0.0)

        # 诊断信息
        nuv_def = result.get("s1_nuv_defect", 0.0)
        scn_def = result.get("s1_scn_defect", 0.0)
        s2b_cls = result.get("s2b_class")
        s2b_cf = result.get("s2b_conf", 0.0)
        crop = result.get("crop_used", False)

        if s1 == "normal":
            msg = (f"[{current}/{total}] {filename}\n"
                   f"    S1: {channel.upper()}正常 | NUV缺陷={nuv_def:.1%} SCN缺陷={scn_def:.1%}")
        elif s2:
            detail = f"S2={s2}({s2_conf:.1%})"
            if s2b_cls:
                detail += f" | S2b={s2b_cls}({s2b_cf:.1%})"
            if crop:
                detail += " | 已裁剪"
            msg = (f"[{current}/{total}] {filename}\n"
                   f"    S1: {channel.upper()}缺陷 | NUV缺陷={nuv_def:.1%} SCN缺陷={scn_def:.1%}\n"
                   f"    {detail} -> {category}")
        else:
            msg = (f"[{current}/{total}] {filename}\n"
                   f"    S1: {channel.upper()} | 分类失败")

        self._log(msg)

    def _on_finished(self, results: list):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.input_btn.setEnabled(True)
        self.output_btn.setEnabled(True)

        total = len(results)
        errors = [r for r in results if "error" in r]
        success = total - len(errors)

        self.status_label.setText(f"处理完成: 成功 {success} 张, 失败 {len(errors)} 张")
        self.progress_bar.setValue(100)

        self._log("-" * 50)
        self._log(f"处理完成! 总计: {total} 张, 成功: {success} 张, 失败: {len(errors)} 张")

        # 统计
        stats = {}
        for r in results:
            cat = r.get("category", "其他")
            stats[cat] = stats.get(cat, 0) + 1

        self._update_result_table(stats)

        if errors:
            self._log(f"失败的文件: {', '.join(r['file'] for r in errors[:5])}")

        QMessageBox.information(
            self,
            "处理完成",
            f"分类完成!\n总计: {total} 张\n成功: {success} 张\n失败: {len(errors)} 张\n\n结果已保存到:\n{self.output_edit.text()}",
        )

    def _on_error(self, error_msg: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.input_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.status_label.setText("处理出错")
        self._log(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{error_msg}")

    def _update_result_table(self, stats: dict):
        rows = sorted(stats.items(), key=lambda x: -x[1])
        self.result_table.setRowCount(len(rows))
        for i, (cat, count) in enumerate(rows):
            self.result_table.setItem(i, 0, QTableWidgetItem(cat))
            item = QTableWidgetItem(str(count))
            item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(i, 1, item)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "确认",
                "正在处理中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def run_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 启动画面：提示用户程序正在加载
    pixmap = QPixmap(480, 240)
    pixmap.fill(QColor("#487eb0"))
    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint)

    font = QFont("Microsoft YaHei", 14, QFont.Bold)
    splash.setFont(font)
    splash.showMessage(
        "晶圆缺陷自动分类系统\n\n正在初始化，请稍候...",
        Qt.AlignCenter,
        QColor("#ffffff"),
    )
    splash.show()
    app.processEvents()

    # 创建主窗口（此过程可能较耗时）
    window = MainWindow()
    window.show()

    splash.finish(window)
    sys.exit(app.exec())
