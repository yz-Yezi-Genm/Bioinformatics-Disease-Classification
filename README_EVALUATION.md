# 模型评估与结果展示

本目录包含随机森林和 SVM 的测试集评估代码、实验结果及图表。

## 查看结果

双击项目根目录的 **`查看模型结果.html`**。

- **随机森林**：本模型的混淆矩阵、ROC、指标表和误分类样本。
- **SVM**：本模型的混淆矩阵、ROC、指标表和误分类样本。
- **两模型对比**：输入特征与实验条件、七项评价指标、TN/FP/FN/TP、ROC-AUC 和误分类样本。

页面和图片均可离线查看，不需要启动服务。图像提供 300 dpi PNG 与 SVG。

完整分析见 `docs/evaluation.md`，指标汇总见 `results/metrics.csv`。数值表保留计算精度，图中按六位小数显示评价指标。

## 运行

在项目根目录、已安装 Python 3.12 的环境中执行：

```bat
python -m pip install -r requirements-evaluation.txt
python -m src.evaluate --output-dir results/check_metrics
python -m src.visualize --output-dir results/check_figures
python -m unittest discover -s tests -v
```

脚本默认读取项目中保存的预测，不需要模型权重。重新生成时使用新的输出目录，避免覆盖已有结果。

## 主要文件

| 文件或目录 | 内容 |
|---|---|
| `src/evaluate.py` | 样本核对、指标计算和基础图表 |
| `src/evaluation_io.py` | 绘图输入检查和通用样式 |
| `src/visualize.py` | 单模型页面和模型对比图 |
| `tests/test_evaluate.py` | 评估计算及输入检查测试 |
| `tests/test_visualization.py` | 绘图输入及输出目录测试 |
| `results/evaluation/` | 对齐预测、误分类、混淆矩阵、ROC 点和运行记录 |
| `results/model_views/` | 分算法展示页面、对比页面与图像 |
| `results/figures/` | 基础混淆矩阵、ROC 和指标对比图 |


## 文件清单

- `README_EVALUATION.md`
- `docs/evaluation.md`
- `requirements-evaluation.txt`
- `results/evaluation/aligned_predictions.csv`
- `results/evaluation/audit.json`
- `results/evaluation/confusion_matrices.csv`
- `results/evaluation/misclassified_samples.csv`
- `results/evaluation/roc_points.csv`
- `results/figures/model_metrics_comparison.png`
- `results/figures/rf_confusion_matrix.png`
- `results/figures/roc_comparison.png`
- `results/figures/svm_confusion_matrix.png`
- `results/metrics.csv`
- `results/model_views/comparison.html`
- `results/model_views/comparison/conditions.png`
- `results/model_views/comparison/conditions.svg`
- `results/model_views/comparison/confusion_matrix.png`
- `results/model_views/comparison/confusion_matrix.svg`
- `results/model_views/comparison/metrics.png`
- `results/model_views/comparison/metrics.svg`
- `results/model_views/comparison/misclassified_samples.png`
- `results/model_views/comparison/misclassified_samples.svg`
- `results/model_views/comparison/roc_curve.png`
- `results/model_views/comparison/roc_curve.svg`
- `results/model_views/display_audit.json`
- `results/model_views/index.html`
- `results/model_views/random_forest.html`
- `results/model_views/random_forest/confusion_matrix.png`
- `results/model_views/random_forest/confusion_matrix.svg`
- `results/model_views/random_forest/metrics.png`
- `results/model_views/random_forest/metrics.svg`
- `results/model_views/random_forest/misclassified_samples.png`
- `results/model_views/random_forest/misclassified_samples.svg`
- `results/model_views/random_forest/roc_curve.png`
- `results/model_views/random_forest/roc_curve.svg`
- `results/model_views/svm.html`
- `results/model_views/svm/confusion_matrix.png`
- `results/model_views/svm/confusion_matrix.svg`
- `results/model_views/svm/metrics.png`
- `results/model_views/svm/metrics.svg`
- `results/model_views/svm/misclassified_samples.png`
- `results/model_views/svm/misclassified_samples.svg`
- `results/model_views/svm/roc_curve.png`
- `results/model_views/svm/roc_curve.svg`
- `src/evaluate.py`
- `src/evaluation_io.py`
- `src/visualize.py`
- `tests/test_evaluate.py`
- `tests/test_visualization.py`
- `查看模型结果.html`
