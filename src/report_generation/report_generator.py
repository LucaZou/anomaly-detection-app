# report_generator.py
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging
from datetime import datetime
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

logger = logging.getLogger('ReportGenerator')

def convert_numpy_types(obj: Any) -> Any:
    """将 numpy 类型转换为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class ReportGenerator:
    """报告生成器，负责分析批量检测结果并生成统计信息和图表"""

    def __init__(self, output_dir: str):
        """
        初始化报告生成器
        
        Args:
            output_dir (str): 报告保存的基础目录
        """
        self.output_dir = os.path.join(output_dir, "reports")
        os.makedirs(self.output_dir, exist_ok=True)
        self.history_file = os.path.join(self.output_dir, "detection_history.json")  # 新增：历史记录文件路径
        self.logger = logger
        self._initialize_history()  # 新增：初始化历史文件

    def _initialize_history(self) -> None:
        """初始化历史记录文件，如果不存在则创建空列表"""
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            self.logger.info(f"初始化历史记录文件: {self.history_file}")

    def save_history(self, report: Dict[str, Any], model_name: str, input_dir: str) -> None:
        """
        保存检测报告到历史记录
        
        Args:
            report (Dict[str, Any]): 报告数据
            model_name (str): 使用的模型名称
            input_dir (str): 输入目录
        """
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # 添加元数据并转换 numpy 类型
            report_entry = {
                "timestamp": report["timestamp"],
                "model_name": model_name,
                "input_dir": input_dir,
                "threshold": report["threshold"],
                "stats": convert_numpy_types(report["stats"]),  # 修改：转换 stats 中的 numpy 类型
                "charts": report["charts"],
                "image_paths": report["image_paths"],  # 新增：保存图像路径
                "scores": report["scores"],           # 新增：保存得分
                "image_count": len(report["image_paths"])
            }
            history.append(report_entry)
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
            self.logger.info(f"历史记录已保存: {report['timestamp']}")
        except Exception as e:
            self.logger.error(f"保存历史记录失败: {str(e)}", exc_info=True)
            raise

    def generate_report(self, image_paths: List[str], scores: List[float], threshold: float,model_name: str = "", input_dir: str = "") -> Dict[str, Any]:
        """
        生成检测报告，包括统计信息和图表
        
        Args:
            image_paths (List[str]): 检测图像路径列表
            scores (List[float]): 异常得分列表
            threshold (float): 异常检测阈值
        
        Returns:
            Dict[str, Any]: 报告数据，包含统计信息和图表路径
        """
        try:
            # 统计信息
            total_images = len(image_paths)
            scores_array = np.array(scores)
            anomaly_count = np.sum(scores_array > threshold)
            anomaly_ratio = anomaly_count / total_images if total_images > 0 else 0
            # 新增：异常程度分级
            mild_threshold = threshold * 1.2  # 示例：轻微异常阈值
            severe_threshold = threshold * 1.5  # 示例：严重异常阈值
            anomaly_levels = {
                "normal": np.sum(scores_array <= threshold),
                "mild": np.sum((scores_array > threshold) & (scores_array <= mild_threshold)),
                "moderate": np.sum((scores_array > mild_threshold) & (scores_array <= severe_threshold)),
                "severe": np.sum(scores_array > severe_threshold)
            }
            stats = {
                "total_images": total_images,
                "anomaly_count": anomaly_count,
                "anomaly_ratio": anomaly_ratio,
                "mean_score": float(np.mean(scores_array)),
                "median_score": float(np.median(scores_array)),
                "std_score": float(np.std(scores_array)),
                "min_score": float(np.min(scores_array)),
                "max_score": float(np.max(scores_array)),
                "anomaly_levels": anomaly_levels  # 新增：异常程度统计
            }

            # 生成图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            histogram_path = self._generate_histogram(scores_array, threshold, timestamp)
            boxplot_path = self._generate_boxplot(scores_array, threshold, timestamp)  # 新增：箱线图

            # 报告数据
            report = {
                "timestamp": timestamp,
                "stats": convert_numpy_types(stats),  # 修改：转换 stats 中的 numpy 类型
                "charts": {"histogram": histogram_path,"boxplot":boxplot_path},
                "image_paths": image_paths,
                "scores": scores,
                "threshold": threshold
            }

            # 新增：保存到历史记录
            self.save_history(report, model_name, input_dir)
            self.logger.info(f"生成报告: 异常图像 {anomaly_count}/{total_images}, "
                           f"平均得分 {stats['mean_score']:.2f}")
            return report

        except Exception as e:
            self.logger.error(f"生成报告失败: {str(e)}", exc_info=True)
            raise

    def _generate_histogram(self, scores: np.ndarray, threshold: float, timestamp: str) -> str:
        """
        生成异常得分的直方图
        
        Args:
            scores (np.ndarray): 异常得分数组
            threshold (float): 异常阈值
            timestamp (str): 时间戳，用于命名文件
        
        Returns:
            str: 图表保存路径
        """
        plt.figure(figsize=(8, 6))
        plt.hist(scores, bins=50, color='skyblue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
        plt.title("Anomaly Score Distribution")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        output_path = os.path.join(self.output_dir, f"histogram_{timestamp}.png")
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        return output_path

    def _generate_boxplot(self, scores: np.ndarray, threshold: float, timestamp: str) -> str:
        """生成异常得分的箱线图"""
        plt.figure(figsize=(8, 6))
        plt.boxplot(scores, vert=False)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
        plt.title("Anomaly Score Boxplot")
        plt.xlabel("Score")
        plt.legend()
        output_path = os.path.join(self.output_dir, f"boxplot_{timestamp}.png")
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        return output_path
    
    def export_to_csv(self, report: Dict[str, Any], filename: str) -> str:
        """
        将报告导出为 CSV 文件
        
        Args:
            report (Dict[str, Any]): 报告数据
            filename (str): CSV 文件名
        
        Returns:
            str: CSV 文件路径
        """
        import pandas as pd
        data = {
            "Image Path": report["image_paths"],
            "Score": report["scores"],
            "Status": ["Anomaly" if s > report["threshold"] else "Normal" for s in report["scores"]]
        }
        df = pd.DataFrame(data)
        output_path = filename if os.path.isabs(filename) else os.path.join(self.output_dir, f"{filename}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保目录存在
        df.to_csv(output_path, index=False)
        self.logger.info(f"报告导出为 CSV: {output_path}")
        return output_path
    

    def export_to_pdf(self, report: Dict[str, Any], filename: str) -> str:
        """将报告导出为 PDF 文件"""
    
        output_path = filename if os.path.isabs(filename) else os.path.join(self.output_dir, f"{filename}.pdf")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保目录存在
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # 添加统计信息
        stats = report["stats"]
        text = (f"<b>Detection Report</b><br/>"
                f"Timestamp: {report['timestamp']}<br/>"
                f"Total Images: {stats['total_images']}<br/>"
                f"Anomaly Count: {stats['anomaly_count']}<br/>"
                f"Anomaly Ratio: {stats['anomaly_ratio']:.2%}<br/>"
                f"Mean Score: {stats['mean_score']:.2f}<br/>"
                f"Anomaly Levels: Normal={stats['anomaly_levels']['normal']}, "
                f"Mild={stats['anomaly_levels']['mild']}, "
                f"Moderate={stats['anomaly_levels']['moderate']}, "
                f"Severe={stats['anomaly_levels']['severe']}")
        story.append(Paragraph(text, styles["Normal"]))
        story.append(Spacer(1, 12))
        
        # 添加图表
        for chart_name, chart_path in report["charts"].items():
            img = Image(chart_path, width=400, height=300)
            story.append(img)
            story.append(Spacer(1, 12))
        
        doc.build(story)
        self.logger.info(f"报告导出为 PDF: {output_path}")
        return output_path