import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

# 读取已经训练好的模型和测试数据
model = joblib.load("models/rf_reproduce_v1.joblib")

with np.load(
    "data/processed/tcga_brca_20261029_v2/test_data.npz",
    allow_pickle=False,
) as data:
    X_test = data["X"]
    y_test = data["y"]
    sample_ids = data["sample_id"]

# 模型只看基因表达，预测类别和肿瘤概率
y_pred = model.predict(X_test)
tumor_probability = model.predict_proba(X_test)[:, 1]

# 将预测与正确答案比较
metrics = {
    "test_samples": len(y_test),
    "accuracy": accuracy_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, tumor_probability),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "tumor_f1": f1_score(y_test, y_pred, pos_label=1),
    "tumor_recall": recall_score(y_test, y_pred, pos_label=1),
    "specificity": recall_score(y_test, y_pred, pos_label=0),
}
matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("测试样本数：", len(y_test))
print(f"Accuracy：{metrics['accuracy']:.6f}")
print(f"ROC-AUC：{metrics['roc_auc']:.6f}")
print(f"Balanced accuracy：{metrics['balanced_accuracy']:.6f}")
print(f"肿瘤 F1：{metrics['tumor_f1']:.6f}")
print(f"肿瘤召回率：{metrics['tumor_recall']:.6f}")
print(f"正常召回率／特异度：{metrics['specificity']:.6f}")

print("\n混淆矩阵：行是真实类别，列是预测类别；顺序均为正常、肿瘤")
print(matrix)

# 保存汇总指标、混淆矩阵和每个样本的预测，便于交接与核对
output_dir = Path("results/rf_reproduce_v1_test")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "metrics.json").write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["actual_class", "predicted_normal", "predicted_tumor"])
    writer.writerow(["normal", *matrix[0]])
    writer.writerow(["tumor", *matrix[1]])

with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["sample_id", "true_label", "predicted_label", "tumor_probability"])
    writer.writerows(zip(sample_ids, y_test, y_pred, tumor_probability))

print("\n结果已保存到：", output_dir.resolve())
