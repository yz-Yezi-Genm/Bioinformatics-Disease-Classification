# TCGA-BRCA 实验一预处理结果

## 用途与限制

任务：Primary Tumor（肿瘤，label=1）与 Solid Tissue Normal（正常，label=0）分类。此交付用于实验一的 Random Forest / SVM 输入准备，不含特征筛选或模型训练。生存表不用于本分类标签，不能把 OS 当成肿瘤/正常标签。

## 数据来源与核验

- 原始压缩包 SHA256：`5caf13eb4ea984cf821ba3fafb94a8ed13ec887c608611d3cac404a62615baa1`
- README 描述表达矩阵为 60,660 个基因 × 1,226 个样本；实际读取尺寸：60,660 × 1,226，样本列名唯一。
- 临床表：1,255 行 × 85 列；按样本 ID 与表达矩阵精确匹配。全部 1,226 个表达样本均找到临床记录；临床表额外的 29 个样本没有表达矩阵，不进入输出。
- 标签取自临床表 `sample_type.samples`，并与 TCGA 条码样本类型代码交叉核验。条码类型计数：`{"01": 1106, "06": 7, "11": 113}`。
- 仅保留 Primary Tumor 与 Solid Tissue Normal。保留 1,219 个样本，类型计数：`{"Primary Tumor": 1106, "Solid Tissue Normal": 113}`。排除类型计数：`{"Metastatic": 7}`。
- 表达矩阵缺失值：0；非有限值：0；负值：0；重复基因 ID：0；重复表达样本 ID：0。
- 全数据共有 2,612 个恒零基因。为保持组员 D 的特征筛选职责，本次保留全部 60,660 个基因。训练集中有 2,648 个恒定特征，已在质控中记录，未据此删列；由 D 在训练折内决定如何过滤。

## 数值与清洗决策

- README 标明表达值为 `log2(count + 1)`，实际数据含小数且范围为 0.0000 至 22.8733。保留原尺度，未取整、未逆变换、未再次 log 转换。该矩阵不能直接作为 DESeq2 原始整数 counts。
- 没有缺失表达值，所以没有进行填补。分类标签/临床信息不拼入模型特征矩阵。
- 输出采用“样本 × 基因”，数组 `X` 为 float32；`y` 为 uint8，0=正常，1=肿瘤。原始表达矩阵未修改。
- SVM 若需要标准化，应在训练折内 fit scaler，再 transform 验证/测试折；随机森林通常不需要标准化。特征选择只能在训练数据或训练交叉验证折中拟合。

## 划分

- 采用 5 折 `StratifiedGroupKFold` 的第 1 折作为约 80/20 训练/测试划分，随机种子 20261029。
- 分组 ID 为 TCGA 病人条码前 12 位；同一患者的所有样本都在同一侧。训练患者数 877，测试患者数 218，交集 0。
- 训练样本数 976，类别计数 `{"normal_0": 88, "tumor_1": 888}`；测试样本数 243，类别计数 `{"normal_0": 25, "tumor_1": 218}`。
- 类别明显不平衡。测试集不能只报告 Accuracy，建议同时由组员 G 报 AUC、F1、敏感度/召回率、特异度、balanced accuracy 与混淆矩阵。模型阈值和超参数在训练数据内部选择，不能根据测试集反复调参。

## 文件

- `train_data.npz`：`X`（样本 × 基因）、`y`、`sample_id`。
- `test_data.npz`：同格式测试数据。
- `feature_ids.tsv`：`X` 列顺序与 Ensembl ID 的对应表，保留版本后缀。
- `sample_manifest.tsv`：样本、病人、临床类型、二分类标签与划分；病人 ID 只用于核查分组，不要作为模型特征。
- `audit.json`：机器可读的统计、版本与校验值。

## Python 读取示例

```python
import numpy as np
train = np.load("train_data.npz", allow_pickle=False)
test = np.load("test_data.npz", allow_pickle=False)
X_train, y_train = train["X"], train["y"]
X_test, y_test = test["X"], test["y"]
feature_ids = open("feature_ids.tsv", encoding="utf-8").read().splitlines()[1:]
```
