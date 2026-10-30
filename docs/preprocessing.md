# TCGA-BRCA 数据预处理说明

本项目的分类任务是用 TCGA-BRCA RNA-seq 表达数据区分原发肿瘤（`label=1`）和实体正常组织（`label=0`）。预处理脚本只负责把 A 提供的表达矩阵和 B 提供的样本标签整理成统一的训练/测试输入；不负责特征选择、随机森林、SVM 或最终评价。

## 输入格式

### 表达矩阵

- TSV 或 TSV.GZ 文件；
- 第一列必须命名为 `Ensembl_ID`，每行一个基因；
- 其余列为样本 ID（TCGA barcode）；
- 数值应为非负且有限；本次 TCGA 文件按来源 README 解释为 `log2(count + 1)`，脚本保持原数值尺度，不再次取对数、不取整。

### 样本标签

默认要求两个字段：

- `sample`：样本 ID，必须与表达矩阵列名一致且唯一；
- `sample_type.samples`：样本类型，使用 `Primary Tumor` 或 `Solid Tissue Normal`。

如果 B 提供的列名不同，可以通过 `--sample-id-column` 和 `--label-column` 指定。表达矩阵中的样本如果没有标签会直接报错，不会静默丢弃。

## 处理流程

1. 读取并检查表达矩阵、标签表的文件格式和列名。
2. 按 `sample_id` 对齐表达样本与临床标签，而不是按文件行号或列号假定顺序一致。
3. 检查重复样本 ID、重复基因 ID、缺失值、非有限值和负值。
4. 用 TCGA barcode 的样本类型代码（可识别时）与标签表交叉核验；不一致时停止。
5. 保留 `Primary Tumor` 和 `Solid Tissue Normal`，排除其他类型（例如 Metastatic）。标签映射为正常 `0`、肿瘤 `1`。
6. 将矩阵从“基因 × 样本”转置为“样本 × 基因”，输出 `float32` 特征和 `uint8` 标签。
7. 按 TCGA 病人 barcode 前 12 位分组，用 `StratifiedGroupKFold` 的第一个折生成约 8:2 的训练/测试划分。同一病人的多个样本不会跨集合。
8. 保留全部基因，不在本步骤做特征筛选。训练集中的恒定特征由 D 在训练折内处理，避免把 D 的职责提前做掉。
9. 输出机器可读的 `audit.json`，记录来源、规模、类别计数、划分、患者重叠数和文件 SHA256。

## 运行命令

在项目根目录执行：

```bash
python src/preprocess.py \
  --expression /path/to/TCGA-BRCA.star_counts.tsv.gz \
  --labels /path/to/TCGA-BRCA.clinical.tsv.gz \
  --output-dir data/processed/tcga_brca_preprocessed
```

若要复现仓库中已经交付的 `v2` 划分，额外指定已有的样本清单：

```bash
python src/preprocess.py \
  --expression /path/to/TCGA-BRCA.star_counts.tsv.gz \
  --labels /path/to/TCGA-BRCA.clinical.tsv.gz \
  --split-manifest data/processed/tcga_brca_20261029_v2/sample_manifest.tsv \
  --output-dir data/processed/tcga_brca_reproduced
```

使用 `--split-manifest` 时，脚本会检查每个样本的 `sample_id`、`label` 和 `split` 是否与标签表一致，因此不依赖不同版本 `scikit-learn` 对随机划分实现的细微差异。未提供清单时，脚本默认使用随机种子 `20261029` 的 `StratifiedGroupKFold` 第一个折。

Windows PowerShell 示例：

```powershell
python .\src\preprocess.py `
  --expression 'D:\data\TCGA-BRCA.star_counts.tsv.gz' `
  --labels 'D:\data\TCGA-BRCA.clinical.tsv.gz' `
  --output-dir '.\data\processed\tcga_brca_preprocessed'
```

脚本拒绝覆盖非空输出目录。建议先在独立目录试运行，再把确认后的结果交付给后续成员。

## 已交付结果

`data/processed/tcga_brca_20261029_v2/` 是本项目已经核验的交付包，包含：

- `train_data.npz` / `test_data.npz`：数组 `X`（样本 × 基因）、`y` 和 `sample_id`；
- `feature_ids.tsv`：特征列序号与 Ensembl ID 的对应关系；
- `sample_manifest.tsv`：样本、患者、样本类型、标签和划分；
- `audit.json`：质量检查、统计和 SHA256；
- `README_preprocessed.md`：本次交付的读取说明和限制。

该版本共保留 60,660 个基因、1,219 个样本；训练集 976 个样本，测试集 243 个样本；训练/测试患者交集为 0。类别不平衡明显，后续不能只报告 Accuracy，应同时报告 AUC、F1、召回率/敏感度、特异度、balanced accuracy 和混淆矩阵。

## 后续成员使用边界

- D：只在训练集或训练交叉验证折上拟合特征筛选，再应用到测试集。
- E：用同一划分训练随机森林。
- F：用同一划分训练 SVM；标准化参数只能在训练数据上拟合。
- G：统一汇总测试集指标和混淆矩阵，不能根据测试集反复调参。

样本 ID、患者 ID 和样本类型只用于核对或分组，不能拼接进模型特征矩阵。原始数据和生存表不属于本交付包；生存信息不用于本次肿瘤/正常分类标签。
