# 数据来源说明

> 负责人：A（数据获取）
> 更新日期：2026-10-30

## 一、数据集信息

| 项 | 内容 |
|---|---|
| 数据集名称 | TCGA-BRCA `STAR - Counts` |
| 所属队列 | GDC TCGA Breast Cancer (BRCA) |
| 数据版本 | GENCODE v36 注释，GDC Hub 发布 |
| 文件格式 | TSV（制表符分隔，原始文件为 gzip 压缩） |
| 样本数量 | 1,226 |
| 基因数量 | 60,660 |
| 矩阵维度 | 60,660 行（基因）× 1,227 列（1 列基因编号 + 1,226 列样本） |
| 表达矩阵 gzip 文件 | `TCGA-BRCA.star_counts.tsv.gz`，138,503,699 字节 |
| 表达矩阵解压后 | `gene_expression.tsv`，754,765,208 字节 |

## 二、下载链接

数据页面（人工获取入口）：

```
https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Breast%20Cancer%20(BRCA)
```

文件直链：

```
https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.star_counts.tsv.gz
https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.clinical.tsv.gz
https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.survival.tsv.gz
https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v36.annotation.gtf.gene.probemap
```

获取方式：网页端进入数据页面，在 `gene expression RNAseq → STAR - Counts` 一行点击 Download；
或在命令行直接请求上述直链。脚本见 `src/fetch_data.py`，可在空目录下一键复现全部文件。

## 三、数据存放位置

```
data/raw/gene_expression.tsv                          表达矩阵（由 gzip 文件解压，60,660 行 × 1,227 列）
data/raw/TCGA-BRCA.star_counts.tsv.gz                 原始 gzip 文件（保留，用于溯源与校验）
data/raw/TCGA-BRCA.clinical.tsv.gz                    临床信息（1,255 行，含样本类型标签）
data/raw/TCGA-BRCA.survival.tsv.gz                    生存信息（1,232 行）
data/raw/gencode.v36.annotation.gtf.gene.probemap     Ensembl ID → 基因名映射（60,660 条）
```

数据文件体积较大，不入 Git 仓库（`.gitignore` 已排除 `data/raw/`），存放于本地。

注：`gencode.v36.annotation.gtf.gene.probemap` 为单独下载的文件，不在表达矩阵的原始 gzip 文件内；`gene_expression.tsv` 由 `TCGA-BRCA.star_counts.tsv.gz` 解压得到。向组员分发时需确保五个文件齐全，不能只发 gzip 文件。

## 四、数据维度统计

矩阵结构：

- 第一列为 Ensembl 基因 ID，带版本号，如 `ENSG00000000003.15`，共 60,660 行
- 其后 1,226 列为样本条码，如 `TCGA-D8-A146-01A`
- 建模前需转置为「样本 × 基因」

数值含义：矩阵内为 **log2(count + 1)**，取值区间 0 ~ 22.87，无缺失值。

样本构成（条码第 14–15 位为类型代码）：

| 代码 | 类型 | 数量 |
|---|---|---|
| 01 | Primary Tumor 原发肿瘤 | 1,106 |
| 11 | Solid Tissue Normal 正常组织 | 113 |
| 06 | Metastatic 转移灶 | 7 |

## 五、数据核验结果

| 核验项 | 方法 | 结果 |
|---|---|---|
| 文件完整性 | `gzip -t` CRC 校验 | 通过 |
| 文件未被改动 | 本地字节数与服务器 `Content-Length` 比对 | 四项全部一致 |
| 文件可读性 | 逐行解析 60,661 行 × 1,227 列 | 行列数一致，无数值解析错误 |
| 数值合法性 | 全矩阵扫描 | 无空值、无负值 |
| 样本数与页面标注 | 与 Xena 页面 `STAR - Counts (n=1,226)` 比对 | 一致 |
| 样本条码真实性 | 与 GDC 官方 API 逐条比对 | 1,226 / 1,226 命中 |
| 样本类型标注 | 条码推断 vs GDC 官方标注 | 0 处不一致 |
| 数据与标签对应 | 标志基因表达量组间比较 | 方向符合已知生物学，无错位 |

## 六、核验的执行方式

上表核验均在数据获取时实际执行，其中只有「文件完整性」与「文件未被改动」由 `src/fetch_data.py` 自动完成（下载后比对字节数），其余为单独执行的检查。复查方式如下。

| 核验项 | 复现方法 |
|---|---|
| gzip 完整性 | `gzip -t data/raw/TCGA-BRCA.star_counts.tsv.gz` |
| 与服务器大小一致 | `stat -c %s` 取本地字节数，`curl -sI <直链>` 取 `Content-Length`，两者对比 |
| 行列数是否一致 | `awk -F'\t' '{print NF}' data/raw/gene_expression.tsv \| sort -u`，结果应为单一值 1227 |
| 空值与负值 | 逐行解析全部单元格，统计 NaN 与小于 0 的个数 |
| 样本条码真实性 | 请求 `https://api.gdc.cancer.gov/cases`（`project.project_id=TCGA-BRCA`，`expand=samples`）取全部 `submitter_id`，与矩阵表头求交集 |
| 样本类型标注 | 条码第 14–15 位与 GDC 返回的 `sample_type` 逐条比对 |
| 数据与标签对应 | 取 KRT5、KRT14、MKI67 等标志基因，按样本类型分组求均值，比较组间方向 |
