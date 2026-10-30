# TCGA-BRCA 标签核对记录

## 核对结果

已将 B 提供的 `sample_labels.csv` 按 `sample_id` 与已经交付的 `data/processed/tcga_brca_20261029_v2/sample_manifest.tsv` 逐条合并核对，并额外用 A 的原始临床表做了交叉检查：

| 检查项 | 结果 |
|---|---:|
| C 清单样本数 | 1,219 |
| B 标签表样本数 | 1,219 |
| 两边共同样本数 | 1,219 |
| C 独有样本 | 0 |
| B 独有样本 | 0 |
| 重复样本 ID（C / B） | 0 / 0 |
| 逐样本标签不一致 | 0 |
| 样本类型不一致 | 0 |
| 患者 ID 不一致 | 0 |
| 原始临床表缺失标签 | 0 |
| 正常组织（label=0） | 113 |
| 原发肿瘤（label=1） | 1,106 |
| 训练集样本 | 976 |
| 测试集样本 | 243 |
| 训练/测试患者交集 | 0 |

标签映射为：

- `Solid Tissue Normal` → `0`；
- `Primary Tumor` → `1`；
- `Metastatic` 不进入本次二分类数据。

B 的文件与 C 的清单行顺序不同，但 `sample_id` 集合完全相同；按样本 ID 合并后，标签、样本类型和患者 ID 均逐项一致。因此不需要重新生成 `train_data.npz`、`test_data.npz` 或 `sample_manifest.tsv`。

## B 标签包的附加信息

B 的 `sample_labels.csv` 包含 `sample_id`、`patient_id`、`sample_type`、`sample_type_code` 和 `label`，标签值只有 `0/1`。B 的 `sample_metadata.csv` 额外标记了 12 个 FFPE 样本；这 12 个样本均为原发肿瘤，且都在当前 1,219 个样本中。当前预处理没有擅自排除 FFPE 样本，是否排除应由组内统一决定。

本次核对检查了：

1. 样本 ID 是否重复；
2. B 文件与 C 清单的交集、缺失样本和多余样本；
3. 逐样本标签、样本类型和患者 ID 是否一致；
4. 正常/肿瘤数量是否一致；
5. C 清单与 A 原始临床表是否一致。

## 交付物存放

- GitHub：`data/processed/tcga_brca_20261029_v2/`，作为仓库中的正式副本；
- 下游组员：发送本地 `TCGA-BRCA-experiment1-preprocessed-20261029-v2.zip`，作为便于下载的副本；
- 两份副本应通过 `audit.json` 中的 SHA256 或压缩包 SHA256 核对，不能分别重新处理出两套划分。

B 标签压缩包 SHA256：`3ac89f07378bc80d2eead75ca8d76725a1616ed9e4e1253866560ccf03253bfb`。
