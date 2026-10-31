# 随机森林：方法、实验与交接

对应 [issue #7](https://github.com/yz-Yezi-Genm/Bioinformatics-Disease-Classification/issues/7)。训练入口现为 `src/random_forest.py`，由原 `src/train_rf.py` 直接重命名并增加补充实验选项，没有保留第二套训练逻辑。原训练记录中的旧命令、路径和哈希保持原样。

## 方法和数据

随机森林对训练样本进行有放回抽样，构建多棵决策树；每次分裂只考虑部分特征，以降低树之间的相关性。分类综合各树的概率，使用 sklearn 默认 predict 决策规则。本任务未调整分类阈值，也没有额外标准化。RF 的 impurity importance 表示模型分裂贡献，受相关特征等因素影响，不等于因果关系。

数据来自 C 的 `TCGA-BRCA-experiment1-preprocessed-20261029-v2.zip`，标签为 0=正常、1=肿瘤。沿用 C 的固定训练／测试划分：976 个训练样本、877 位训练患者，独立测试预测记录有 243 个样本。没有重新划分数据。

训练和补充分析只读取以下四个输入：

- `data/processed/tcga_brca_20261029_v2/train_data.npz`
- `data/processed/tcga_brca_20261029_v2/feature_ids.tsv`
- `results/feature_selection/cv_fold_assignments.csv`
- `results/feature_selection/cv_metadata.json`

自编精简入口不调用 D 的 `load_training_data`，不读取审计文件、完整样本清单或原始标签表。入口集中检查维度、有限值、二分类标签、样本唯一性、原始特征索引和 Ensembl ID；训练文件哈希与 D 记录一致。patient_id 取 sample_id 前 12 位，按样本核对折记录、标签和患者，各患者只属于一个验证折，训练／验证部分均含两类。

直接将 D 的 `create_selector(k=100)` 接入 Pipeline，再接 RF。原始 60,660 列及其顺序保持不变，样本 ID、患者 ID 和标签不进入 X。每折克隆新 Pipeline，只在该训练折拟合方差过滤和 ANOVA 筛选器，验证折仅 transform，避免把验证标签用于筛选。没有提前使用完整训练基因名单，未修改 C、D 的代码和数据。

## 原训练与内部选模

使用 D 保存的患者五折，固定种子 20261031。主要选模指标预先固定为五折验证 ROC-AUC 均值，并列取预定义候选顺序中的第一项。另报 balanced accuracy、肿瘤 F1、肿瘤召回率、正常召回率（特异度）。均值为五折算术均值，标准差为样本标准差（ddof=1），不是置信区间。

原实验含基础 RF 在内共 10 组，`n_estimators=200`、`min_samples_split=2` 固定，其他参数与真实 AUC 如下（None 表示 sklearn 的对应默认含义）：

| candidate | max_depth | min_samples_leaf | max_features | class_weight | AUC mean |
|---|---|---|---|---|---|
| rf_00 | None | 1 | sqrt | balanced_subsample | 0.999844 |
| rf_01 | None | 1 | sqrt | None | 0.999746 |
| rf_02 | None | 2 | sqrt | balanced | 0.999813 |
| rf_03 | 12 | 4 | sqrt | balanced_subsample | 0.999813 |
| rf_04 | None | 1 | 0.3 | balanced_subsample | 0.999813 |
| rf_05 | 12 | 2 | 0.3 | balanced | 0.999813 |
| rf_06 | None | 4 | 0.3 | None | 0.999688 |
| rf_07 | 8 | 1 | 0.5 | balanced | 0.999750 |
| rf_08 | None | 2 | 0.5 | balanced_subsample | 0.999750 |
| rf_09 | 8 | 2 | sqrt | None | 0.999750 |

候选 0 即基础模型，也是原训练的最佳候选。原实验完成 50 次 RF 折拟合、5 次多数类基线折拟合及一次完整训练拟合，约用时 86.2 秒。有限搜索未提高主要指标。所有 CV 成绩均为**训练集内部选模结果**，不能称为独立测试成绩或无偏泛化估计。

| 内部五折指标 | 多数类基线均值 | 原基础／冻结 RF 均值 ± 标准差 |
|---|---:|---:|
| ROC-AUC | 0.500000 | 0.999844 ± 0.000349 |
| Balanced accuracy | 0.500000 | 0.965451 ± 0.029521 |
| 肿瘤 F1 | 0.952792 | 0.996078 ± 0.002479 |
| 肿瘤召回率 | 1.000000 | 0.998876 ± 0.002512 |
| 特异度 | 0.000000 | 0.932026 ± 0.060585 |

多数类基线虽有较高肿瘤 F1，却完全不能识别正常样本，因此需结合特异度和 balanced accuracy。

## 独立测试后的补充参数分析

本补充实验发生在独立测试已经完成、成绩已经可见之后，**仅用于参数分析，不替换原冻结模型，不重新选优**。每个配置仅相对基础配置改变一个参数：树数 100、400，或 min_samples_split 为 5、10。固定 k=100、随机种子、患者折、评分规则和默认分类决策。每折重新拟合 D 的筛选器，未运行原 10 组搜索，也未读取测试表达矩阵。

本次环境、四个训练输入哈希、D 源码及基础配置与原实验一致，因此复用原基础五折对照，实际新增 20 次 RF 折拟合。候选和折顺序执行，RF 为 2 线程、数值库限制为 2 线程，没有两层全核并行。

| candidate | ROC-AUC | Balanced accuracy | Tumor F1 | Tumor recall | Specificity |
|---|---|---|---|---|---|
| baseline | 0.999844 +/- 0.000349 | 0.965451 +/- 0.029521 | 0.996078 +/- 0.002479 | 0.998876 +/- 0.002512 | 0.932026 +/- 0.060585 |
| trees_100 | 0.999875 +/- 0.000279 | 0.965451 +/- 0.029521 | 0.996078 +/- 0.002479 | 0.998876 +/- 0.002512 | 0.932026 +/- 0.060585 |
| trees_400 | 0.999813 +/- 0.000419 | 0.965451 +/- 0.029521 | 0.996078 +/- 0.002479 | 0.998876 +/- 0.002512 | 0.932026 +/- 0.060585 |
| split_5 | 0.999750 +/- 0.000407 | 0.965451 +/- 0.029521 | 0.996078 +/- 0.002479 | 0.998876 +/- 0.002512 | 0.932026 +/- 0.060585 |
| split_10 | 0.999750 +/- 0.000407 | 0.965451 +/- 0.029521 | 0.996078 +/- 0.002479 | 0.998876 +/- 0.002512 | 0.932026 +/- 0.060585 |

补充配置的四项分类指标均与原基础配置相同。100 棵树的平均 AUC 比对照高约 0.000031，其余补充配置略低；差异很小，不能据此声称显著优势。冻结模型仍为原 200 棵树配置，未因补充结果改变参数、特征数或阈值。

## 冻结模型与独立测试

`models/rf_pipeline.joblib` 包含筛选器与 RF，在全部 976 个训练样本上拟合。完整参数来自实际保存模型：

```json
{
  "selector": {
    "k": 100,
    "variance_threshold": 0.0
  },
  "classifier": {
    "bootstrap": true,
    "ccp_alpha": 0.0,
    "class_weight": "balanced_subsample",
    "criterion": "gini",
    "max_depth": null,
    "max_features": "sqrt",
    "max_leaf_nodes": null,
    "max_samples": null,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "monotonic_cst": null,
    "n_estimators": 200,
    "n_jobs": 2,
    "oob_score": false,
    "random_state": 20261031,
    "verbose": 0,
    "warm_start": false
  }
}
```

冻结模型 SHA256：`f2a231ddc814803334770447590d622c378c76c5d35ad919d3ada8bf045a7b4d`。本轮开始前和完成后均核对该值，模型未改变。原训练记录和原测试结果的文件哈希也保持一致。

独立测试已由保留的 `test_rf_manual.py` 完成。本轮只从已有 `results/rf_test_manual/predictions.csv` 重算指标，没有再次加载测试表达矩阵或重复运行测试。核验了 243 个唯一且非空 sample_id，真实标签、预测标签为 0/1，概率有限且在 [0,1] 内，四个字段齐全。

| 独立测试指标 | 数值 |
|---|---:|
| Accuracy | 0.991770 |
| ROC-AUC | 0.999725 |
| Balanced accuracy | 0.977706 |
| 肿瘤 F1 | 0.995413 |
| 肿瘤召回率 | 0.995413 |
| 特异度 | 0.960000 |

混淆矩阵为 `[[24, 1], [1, 217]]`，行是真实类别、列是预测类别，顺序为正常／肿瘤。结果与原指标及混淆矩阵文件一致。正常测试样本只有 25 个，每错一个会使特异度变化 0.04；不能把本次较高成绩外推为稳定的临床性能，仍需更广泛的独立验证。

## 命令、输出与依赖

在项目根目录运行；输出目录和模型路径必须是新的，入口拒绝覆盖既有结果。以下示例使用未占用路径，若已运行过请再换名称：

```powershell
python -m src.random_forest --help
python -m src.random_forest --output-dir results/rf_reproduce_v1 --model-path models/rf_reproduce_v1.joblib --jobs 2
python -m src.random_forest --supplement --output-dir results/rf_supplement_reproduce_v1 --jobs 2
```

第一条检查命令行；第二条重现原训练方案并产生新模型，第三条仅复现四组补充分析、不保存模型。本轮只运行了补充分析，没有重新训练冻结模型。

已有独立测试的命令为 `python test_rf_manual.py`；该脚本固定读取 `models/rf_pipeline.joblib` 和同版本的 `data/processed/tcga_brca_20261029_v2/test_data.npz`，输出到 `results/rf_test_manual/`，再次执行会覆盖这三个原结果文件。本轮没有执行此命令；仅记录供 G 在明确需要重现测试时使用。模型加载需要可导入的 `src.feature_selection`。

- `results/rf/`：原参数搜索、基础与多数类基线、折外预测、实际折分配、运行配置、输入哈希、选中基因与重要性、原核验记录。历史记录不修改。
- `results/rf_supplement/`：四组补充实验及基础对照的折指标、五项汇总、折外预测、实际折分配、运行环境、冻结模型完整参数、本轮核验记录和保护文件哈希。
- `results/rf_test_manual/`：保留原 `metrics.json`、`confusion_matrix.csv`、`predictions.csv`。
- `requirements-rf.txt`：实际依赖为 Python 3.13.9，NumPy 2.3.5、pandas 2.3.3、SciPy 1.16.3、scikit-learn 1.7.2、joblib 1.5.2、threadpoolctl 3.5.0。本轮继续使用现有环境，没有升级或修改团队依赖文件。
- 模型受现有 Git 忽略规则保护，不提交模型、数据矩阵、压缩包、解压目录或缓存。

## 与 F、G 的交接约定

F 应使用 C 同版本的 `train_data.npz` 和 `test_data.npz`，不自行重新划分；建议沿用 D 的内部患者五折，SVM 的特征筛选和标准化都放入 Pipeline，在每个训练折内拟合。不能把完整训练集选中的基因提前用于内部 CV。上述为需要 F 沿用的约定，尚未与 F 确认一致。

G 接收原冻结模型、完整特征映射、依赖及参数记录，以及 RF 原独立测试的逐样本预测、指标和混淆矩阵，按 sample_id 与 F 的同版本预测对齐后统一评价。需由组长确认交付范围，F 确认数据版本和划分，G 确认两模型预测样本一致及报告口径。补充结果单列为测试后分析，不用它替换冻结模型或其独立测试成绩。
