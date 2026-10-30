# TCGA-BRCA 标签核对记录

## 当前可复核结果

在收到 B 的最终标签文件前，先用 A 提供的 `TCGA-BRCA.clinical.tsv.gz` 对已经交付的 `data/processed/tcga_brca_20261029_v2/sample_manifest.tsv` 做了内部一致性检查：

| 检查项 | 结果 |
|---|---:|
| 预处理清单样本数 | 1,219 |
| 在临床表中找到的样本 | 1,219 |
| 缺失临床标签 | 0 |
| 标签不一致 | 0 |
| 正常组织（label=0） | 113 |
| 原发肿瘤（label=1） | 1,106 |
| 训练集样本 | 976 |
| 测试集样本 | 243 |
| 训练/测试患者交集 | 0 |

标签映射为：

- `Solid Tissue Normal` → `0`；
- `Primary Tumor` → `1`；
- `Metastatic` 不进入本次二分类数据。

## 尚待 B 确认的内容

Issue 要求将 A 的表达矩阵与 B 最终整理的标签文件逐个 `sample_id` 核对。目前仓库和本地工作目录中没有发现 B 单独提交的标签文件，因此上面的结果只能证明当前交付包与原始临床表一致，不能替代与 B 文件的交叉核对。

B 只需提供最终标签表的仓库路径或文件，至少包含：

```text
sample_id    label
```

其中 `label` 必须明确约定为正常 `0`、肿瘤 `1`。收到后应检查：

1. 样本 ID 是否重复；
2. B 文件与表达矩阵的交集、缺失样本和多余样本；
3. 逐样本标签是否一致；
4. 正常/肿瘤数量是否一致；
5. 是否需要按 B 的最终标签重新生成 `train_data.npz`、`test_data.npz` 和 `sample_manifest.tsv`。

## 交付物存放

- GitHub：`data/processed/tcga_brca_20261029_v2/`，作为仓库中的正式副本；
- 下游组员：发送本地 `TCGA-BRCA-experiment1-preprocessed-20261029-v2.zip`，作为便于下载的副本；
- 两份副本应通过 `audit.json` 中的 SHA256 或压缩包 SHA256 核对，不能分别重新处理出两套划分。
