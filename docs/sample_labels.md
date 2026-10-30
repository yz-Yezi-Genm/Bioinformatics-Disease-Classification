# TCGA-BRCA 样本标签整理说明（角色 B）

> 模块：B · 标签整理
> 交付物：样本注释表 `sample_metadata.csv`、二分类标签表 `sample_labels.csv`
> 数据集：GDC TCGA Breast Cancer (BRCA)，经 UCSC Xena 整理分发
> 数据版本：成员 A 提供的原始资源包（下载日期 2026-10-29），**与 A 完全同源、同版本**

---

## 1. 任务与结论速览

从 TCGA-BRCA 中为每个 RNA-seq 样本确定组织类型，构建「肿瘤 vs 正常」二分类标签（Tumor = 1，Normal = 0），并确认标签样本 ID 与 A 的表达矩阵可以严格匹配。

| 指标 | 数值 |
|---|---|
| 临床/注释表样本数 | 1255 |
| A 的表达矩阵样本数 | 1226（60660 基因 × 1226 样本） |
| 与表达矩阵成功匹配的样本 | **1226**（A 的样本 100% 在注释表中找到） |
| 二分类可用样本 | **1219**（肿瘤 1106 + 正常 113） |
| 剔除样本 | 7（转移灶 Metastatic，代码 06） |
| 肿瘤 : 正常 | 1106 : 113 ≈ **9.79 : 1** |
| 重复样本 ID | 0 |
| 条码类型码与临床字段冲突 | 0（1255/1255 完全一致） |

---

## 2. 数据来源

| 项目 | 内容 |
|---|---|
| 数据集 | TCGA Breast Invasive Carcinoma（TCGA-BRCA） |
| 数据门户 | UCSC Xena（GDC TCGA Breast Cancer, BRCA cohort） |
| 下载日期 | 2026-10-29 |
| 提供者 | 成员 A（群内共享原始资源包 `TCGA-BRCA-data`） |
| 文件 1 | `TCGA-BRCA.clinical.tsv.gz`：样本临床/注释表，1255 行 × 85 列 |
| 文件 2 | `TCGA-BRCA.star_counts.tsv.gz`：RNA-seq 表达矩阵，60660 基因 × 1226 样本，值为 log2(count+1) |
| 文件 3 | `TCGA-BRCA.survival.tsv.gz`：生存信息（OS.time / OS / _PATIENT），本次未使用 |
| 标签字段 | `sample_type.samples`（第 74 列）与 `sample_type_id.samples`（第 71 列），辅助核对字段 `tumor_descriptor.samples` |

说明：A 的表达矩阵即本任务的「样本全集」基准。凡是 A 矩阵中出现的样本，都会出现在 `sample_metadata.csv` 中，其中肿瘤/正常样本进入 `sample_labels.csv`，转移灶样本进入剔除清单，不存在无标签却被静默丢弃的样本。

---

## 3. 样本 ID 命名规则

TCGA 样本条码为 16 位：`TCGA-XX-XXXX-01A`

| 片段 | 示例 | 含义 |
|---|---|---|
| `TCGA` | `TCGA` | 项目代号 |
| `XX` | `BH` | 组织来源中心（Tissue Source Site, TSS） |
| `XXXX` | `A0W3` | 患者编号（Participant） |
| `01` | `01` | **样本类型代码（第 14–15 位）** |
| `A` | `A` | 样本顺序/瓶号（vial），同一患者同类型样本可能为 A、B、C |

- 患者 ID = 条码前 12 位（`TCGA-BH-A0W3`），已逐行与临床表 `submitter_id` 比对，**0 处不一致**。
- 本数据集全部样本均为 16 位，未出现 aliquot 级后缀（如 `-01A-01R`），因此可以直接按 16 位条码精确匹配。
- 本队列实际出现的瓶号为 A（1180 个）、B（38 个）、C（1 个）。

### 样本类型代码

| 代码 | 英文类型 | 中文 | 样本数（临床表 / 表达矩阵） | 本任务处理 |
|---|---|---|---|---|
| `01` | Primary Tumor | 原发肿瘤 | 1111 / 1106 | **Tumor = 1** |
| `11` | Solid Tissue Normal | 癌旁正常组织（实性组织正常） | 137 / 113 | **Normal = 0** |
| `06` | Metastatic | 转移灶 | 7 / 7 | **剔除**，单独记录 |

> 本队列只出现 01、11、06 三种代码；02（复发灶）、03（血液来源肿瘤）、07（额外转移灶）等其他代码均未出现。

---

## 4. 标签定义

```
label = 1  当且仅当 sample_type.samples == "Primary Tumor"（代码 01）
label = 0  当且仅当 sample_type.samples == "Solid Tissue Normal"（代码 11）
label = 空 其他类型（本次为 Metastatic，代码 06），不纳入二分类
```

- **双重来源核对**：标签同时由「条码第 14–15 位类型码」与「临床表 `sample_type.samples` / `sample_type_id.samples`」两条独立路径推导，1255 条记录**逐条一致（0 冲突）**，因此标签不依赖单一路径。
- **不做主观归类**：`Metastatic` 样本不会被强行并入肿瘤组；按任务要求单列在 `results/tables/excluded_samples.csv`。
- **Normal 的准确含义**：`Solid Tissue Normal` 是同一患者的癌旁正常乳腺组织（adjacent normal），不是健康人群对照。写论文/报告时应表述为「tumor vs adjacent normal」，不要写成「tumor vs healthy」。

---

## 5. 交付文件

### 5.1 `data/raw/sample_labels.csv`（二分类标签表，1219 行）

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | str | TCGA 16 位样本条码，与 A 的表达矩阵列名一一对应 |
| `patient_id` | str | 患者 ID（条码前 12 位） |
| `sample_type` | str | `Primary Tumor` / `Solid Tissue Normal` |
| `sample_type_code` | str | `01` / `11` |
| `label` | int | 1 = 肿瘤，0 = 正常 |

该表只含二分类样本，可直接与表达矩阵 merge，无需再做过滤。

### 5.2 `data/raw/sample_metadata.csv`（样本注释表，1226 行）

在标签表 5 个字段的基础上，追加患者层面信息与建模辅助字段：

| 字段 | 说明 |
|---|---|
| `tumor_descriptor` | Primary / Not Applicable / Metastatic |
| `specimen_type` | 标本类型（Solid Tissue 等） |
| `tissue_type` | Tumor / Normal |
| `is_ffpe` | 是否 FFPE 蜡块样本（True/False） |
| `gender` | 性别 |
| `age_at_index` | 初诊年龄（岁） |
| `vital_status` | 生存状态 |
| `ajcc_pathologic_stage` | AJCC 病理分期 |
| `primary_diagnosis` | 病理诊断 |
| `days_to_death` / `days_to_last_follow_up` | 生存随访天数 |
| `label` | 0/1；Metastatic 行为空 |
| `in_binary_task` | True/False，标记该样本是否进入二分类 |

### 5.3 统计与核查结果

| 文件 | 内容 |
|---|---|
| `results/tables/sample_type_counts.csv` | 各类型在临床表与表达矩阵中的数量、对应标签、是否纳入 |
| `results/tables/patient_level_summary.csv` | 患者数、配对样本数、多样本患者数等 |
| `results/tables/excluded_samples.csv` | 7 个转移灶样本清单及剔除原因 |
| `results/tables/clinical_not_in_expression.csv` | 29 个有临床记录但不在 A 表达矩阵中的样本 |
| `results/stats.json` | 全部核查指标的机器可读版本 |

---

## 6. 统计结果

### 6.1 样本层面

| 样本类型 | 临床表 | A 的表达矩阵 | 二分类 |
|---|---|---|---|
| Primary Tumor（01） | 1111 | 1106 | 1106（label = 1） |
| Solid Tissue Normal（11） | 137 | 113 | 113（label = 0） |
| Metastatic（06） | 7 | 7 | 0（剔除） |
| **合计** | **1255** | **1226** | **1219** |

### 6.2 样本 ID 对齐

- A 的表达矩阵共 1226 个样本列，全部能在临床表中找到对应行：**表达矩阵独有、临床表缺失的样本 = 0**。
- 临床表 1255 行中有 29 行在 A 的矩阵中没有表达数据（24 个正常组织 + 5 个肿瘤），原因是这些样本未进入该表达矩阵版本。为避免 C 合并时出现空行，这 29 个样本**不写入** `sample_metadata.csv`，单独列在 `clinical_not_in_expression.csv`。
- 匹配方式：按 16 位 `sample_id` 精确匹配（非按行号、非按顺序），匹配成功率 100%。

### 6.3 患者层面

| 指标 | 数值 |
|---|---|
| 分析队列患者数 | 1095 |
| 同时拥有肿瘤与正常样本的患者 | 113 |
| 仅有正常样本、无肿瘤样本的患者 | 0 |
| 含 2 个及以上肿瘤样本的患者 | 11 |
| 含 2 个及以上二分类样本的患者 | 120 |
| 每患者样本数（1 / 2 / 3） | 971 / 117 / 7 |

二分类队列的患者构成：

| 样本构成 | 患者数 | 样本数 |
|---|---|---|
| 仅 1 个肿瘤样本 | 975 | 975 |
| 1 肿瘤 + 1 正常 | 109 | 218 |
| 2 个肿瘤样本 | 7 | 14 |
| 2 肿瘤 + 1 正常 | 4 | 12 |
| **合计** | **1095** | **1219** |

7 个转移灶样本来自 7 位患者，这 7 位患者均另有原发肿瘤样本保留在二分类队列中，因此剔除转移灶**不会导致任何患者整体丢失**（其中 4 位患者剔除后只剩 1 个肿瘤样本，故二分类队列的单样本患者为 975 位，比分析队列的 971 位多 4 位）。

### 6.4 类别不平衡

肿瘤 : 正常 = 1106 : 113 ≈ **9.79 : 1**。全部预测为肿瘤即可得到约 90.7% 的准确率，因此：

- 评估必须报告 **AUC、F1、召回率/特异度**，不能只看 Accuracy；
- 若做过采样/加权，**只能在训练集内做**，绝不能先对全体样本过采样再划分（会造成同一患者样本泄漏）；
- 建议使用 `class_weight="balanced"` 或 `StratifiedGroupKFold(groups=patient_id)`。

---

## 7. 质量核查（QC）

| 检查项 | 方法 | 结果 |
|---|---|---|
| 重复样本 ID | 对 `sample` 列计数 | 0 个重复（1255 个 ID 全部唯一） |
| 缺失组织类型 | 检查 `sample_type.samples` 是否为空 | 0 个缺失 |
| 条码类型码 vs 临床类型码 | `sample_id[13:15]` 与 `sample_type_id.samples` 比对 | 0 处冲突 |
| 条码类型码 vs 临床类型名 | 与 `sample_type.samples` 比对 | 0 处冲突 |
| 患者 ID 一致性 | 条码前 12 位与 `submitter_id` 比对 | 0 处冲突 |
| 无法确定类型的样本 | — | 无，全部样本均有明确的 01/11/06 代码 |
| 标签缺失 | 二分类表内检查 | 0（7 个空标签即被剔除的转移灶，已单列） |
| 患者信息缺失 | 空值统计 | `age_at_index` 缺失 1、`vital_status` 缺失 1、`ajcc_pathologic_stage` 缺失 13（不影响标签） |
| 与 A 的样本匹配 | 与表达矩阵表头取交集 | 1226/1226 = 100% |
| FFPE 样本 | `is_ffpe` 字段 | 12 个，均为肿瘤样本；本表**未擅自剔除**，已用 `is_ffpe` 标记，C 若需可自行过滤 |

---

## 8. 给下游（C / E / F）的使用方式

```python
import pandas as pd

labels = pd.read_csv("data/raw/sample_labels.csv", dtype={"sample_id": str})
expr = pd.read_csv("data/processed/expression.tsv", sep="\t", index_col=0)  # C 转置后：行=样本

df = expr.join(labels.set_index("sample_id"), how="inner")
assert len(df) == 1219, len(df)                       # 样本数核对
assert df["label"].isin([0, 1]).all()                 # 标签只含 0/1
assert df.index.is_unique                             # 无重复样本
print(df["label"].value_counts())                     # 1106 / 113
```

要点：

1. **按 `sample_id` 合并，不要按行号**。临床表行序与表达矩阵列序不同。
2. 标签列是整数 0/1，不要当字符串处理；`sample_id` 建议显式用 `dtype={"sample_id": str}` 读取，避免后续按 ID 合并时类型不一致。
3. **按患者划分数据集**：`GroupShuffleSplit` / `StratifiedGroupKFold`，`groups = patient_id`。113 位患者同时有肿瘤和正常样本、11 位患者有 2 个肿瘤样本，若按样本随机划分，同一患者的样本会分处训练/测试集，AUC 会被高估。
4. 若 A 后续导出的表达矩阵列名变成 15 位（去掉瓶号 A/B/C）或带 aliquot 后缀，用前 15 位做匹配：

```python
labels["key"] = labels["sample_id"].str[:15]
expr.columns = expr.columns.str[:15]
```

5. 转移灶样本（`excluded_samples.csv`）如需用于「转移 vs 原发」的补充分析，请单独建任务，不要混入二分类。

---

## 9. 复现方式

```bash
python src/build_labels.py --raw "C:/Users/asus/Downloads/TCGA-BRCA-data"
```

脚本只读取 A 资源包中的 `TCGA-BRCA.clinical.tsv.gz` 与 `TCGA-BRCA.star_counts.tsv.gz`（后者仅读表头），重新生成 `data/raw/*.csv` 与 `results/*`，不修改原始数据。运行环境：Python 3 + 标准库，无第三方依赖。

---

## 10. 与 A 的接口确认（交付前必读）

| 检查 | 结果 |
|---|---|
| 数据集与版本一致 | 是，均来自 A 的 `TCGA-BRCA-data` 资源包 |
| 表达矩阵样本数 | 1226 |
| 标签可匹配样本数 | 1226（100%） |
| 二分类建模样本数 | 1219 |
| 条码格式 | 16 位 `TCGA-XX-XXXX-01A`，无 aliquot 后缀 |

结论：B 的标签表与 A 的表达矩阵在样本 ID 层面**完全对齐**，可直接进入 C 的预处理流程；7 个转移灶样本已单列剔除，未混入二分类标签。
