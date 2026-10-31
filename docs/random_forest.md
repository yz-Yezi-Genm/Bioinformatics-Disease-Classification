# 随机森林分类实验

本实验对应 [issue #7](https://github.com/yz-Yezi-Genm/Bioinformatics-Disease-Classification/issues/7)，区分 TCGA-BRCA 的原发肿瘤与实体正常组织。实验先在训练集内部比较参数，在完整训练集上拟合并冻结模型后完成独立测试；测试后补做了四组参数分析，但没有改变冻结模型或原测试结果。

## 1. 数据与划分

原始数据来自 **UCSC Xena 的 TCGA-BRCA RNA-seq 数据集（GDC Hub，STAR - Counts）**，详见项目的[数据来源说明](data_sources.md)。C 提供的 `TCGA-BRCA-experiment1-preprocessed-20261029-v2.zip` 是预处理数据的交付副本，不是原始数据来源。

沿用 C 已确定的训练／测试划分，标签为 **0=实体正常组织，1=原发肿瘤**：

| 集合 | 正常 | 肿瘤 | 合计 |
|---|---:|---:|---:|
| 训练集 | 88 | 888 | 976 |
| 独立测试集 | 25 | 218 | 243 |

每个样本包含 60,660 个基因。依据 C 的说明，表达值按 `log2(count + 1)` 尺度保留，RF 不再次取对数，也不额外标准化。C 按患者分组划分数据，记录训练／测试患者交集为 0；这一结论依据 C 的[预处理说明](preprocessing.md)和[交付读取说明](../data/processed/tcga_brca_20261029_v2/README_preprocessed.md)，不是 RF 实验重新核验原始数据所得。

训练和补充分析使用以下输入：

- `data/processed/tcga_brca_20261029_v2/train_data.npz`
- `data/processed/tcga_brca_20261029_v2/feature_ids.tsv`
- `results/feature_selection/cv_fold_assignments.csv`
- `results/feature_selection/cv_metadata.json`

训练入口检查样本、标签、基因列顺序及训练文件哈希，并按 sample_id 对齐 D 的内部折记录。patient_id 取样本条码前 12 位，确保同一患者不跨内部验证折，每个训练折和验证折均包含两类。样本 ID、患者 ID 和标签不作为表达特征。

## 2. 方法与模型参数

随机森林对训练样本有放回抽样，构建多棵决策树；每次分裂只考虑部分特征，以降低树之间的相关性。模型汇总各树的类别概率，使用 sklearn 默认分类规则，不另行调整阈值。

特征选择直接复用 D 的 `create_selector(k=100)`：**先去除当前训练数据中取值恒定的基因，再根据 ANOVA F 得分选择前 100 个基因**。F 得分衡量两类之间的表达差异相对于类内波动的大小。k=100 沿用 D 的[训练集内部参考建议](feature_selection.md)，原 RF 参数搜索固定该数量，没有重新搜索 k，也不能据此认定 100 个基因全局最优。

筛选器与 RF 放在同一个 Pipeline 中。每个训练折重新拟合筛选器，验证折只应用已经学到的规则，避免验证标签参与筛选。Pipeline 始终接收原始顺序的 60,660 列，不提前按完整训练集的基因名单裁剪。

冻结模型的关键参数如下，完整参数见[参数记录](../results/rf_supplement/frozen_model_parameters.json)。

| 参数 | 冻结值 | 含义 |
|---|---|---|
| k | 100 | 筛选器保留的基因数量。 |
| n_estimators | 200 | 随机森林包含 200 棵树。 |
| max_depth | None | 不显式限制树深，仍受其他停止条件约束。 |
| min_samples_split | 2 | 一个节点至少有 2 个样本才允许继续分裂。 |
| min_samples_leaf | 1 | 每个叶节点至少保留 1 个训练样本。 |
| max_features | sqrt | 每次分裂考虑约为输入特征数平方根的候选特征，100 个基因时为 10 个。 |
| class_weight | balanced_subsample | 根据每棵树抽样后的类别数量设置权重，减轻类别不平衡的影响。 |
| random_state | 20261031 | 固定随机过程，便于复现。 |

## 3. 参数比较与内部验证

沿用 D 保存的患者五折，参考种子为 20261031。主要选模指标预先固定为五折验证 ROC-AUC 均值，完全并列时取预定义候选顺序中的第一项；同时记录 balanced accuracy、肿瘤 F1、肿瘤召回率和特异度（正常召回率）。以下标准差采用五折样本标准差（ddof=1），不是置信区间。

### 原参数比较

共比较 10 组配置，固定 `n_estimators=200`、`min_samples_split=2`。表中 `max_depth=None` 表示不显式限制树深，`class_weight=None` 表示不额外设置类别权重。

| 配置 | max_depth | min_samples_leaf | max_features | class_weight | 平均验证 AUC |
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

基础配置 rf_00 获得最高平均验证 AUC，随后在完整训练集上拟合并冻结。各候选的五项指标见[原参数比较结果](../results/rf/parameter_search.csv)。这次有限搜索未提高基础配置的主要指标。

多数类基线在每个训练折中拟合，只预测该折训练数据的多数类。其结果与冻结配置的内部验证结果如下：

| 内部五折指标 | 多数类基线均值 | rf_00 均值 ± 标准差 |
|---|---:|---:|
| ROC-AUC | 0.500000 | 0.999844 ± 0.000349 |
| Balanced accuracy | 0.500000 | 0.965451 ± 0.029521 |
| 肿瘤 F1 | 0.952792 | 0.996078 ± 0.002479 |
| 肿瘤召回率 | 1.000000 | 0.998876 ± 0.002512 |
| 特异度 | 0.000000 | 0.932026 ± 0.060585 |

多数类基线虽有较高肿瘤 F1，却不能识别正常样本。这些 CV 成绩用于训练集内部选模，不是独立测试成绩或无偏泛化估计。

### 测试后的补充分析

为补齐树数和节点分裂条件的比较，分别将基础配置的 `n_estimators` 改为 100、400，或将 `min_samples_split` 改为 5、10。每组只改变一个参数，保持 k、患者五折、随机种子、评分规则和折内筛选方式不变。环境、训练输入和基础配置与原实验一致，因此复用原基础五折对照。

| 配置 | 唯一变化 | 验证 AUC 均值 ± 标准差 |
|---|---|---:|
| baseline | 原基础配置 | 0.999844 ± 0.000349 |
| trees_100 | n_estimators=100 | 0.999875 ± 0.000279 |
| trees_400 | n_estimators=400 | 0.999813 ± 0.000419 |
| split_5 | min_samples_split=5 | 0.999750 ± 0.000407 |
| split_10 | min_samples_split=10 | 0.999750 ± 0.000407 |

四组的 balanced accuracy、肿瘤 F1、肿瘤召回率和特异度均与上表 rf_00 相同，完整结果见[补充分析汇总](../results/rf_supplement/cv_summary.csv)。100 棵树的平均 AUC 比对照高约 0.000031，差异很小，不能据此声称显著优势。由于补充分析发生在独立测试之后，它只用于参数分析，不作为重新选择模型的依据；最终模型仍为原 200 棵树配置。

## 4. 独立测试结果

冻结模型在 C 保留的 243 个独立测试样本上得到以下结果，记录见[测试指标](../results/rf_test_manual/metrics.json)。测试成绩没有用于调整参数、特征数量或分类阈值。

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.991770 |
| ROC-AUC | 0.999725 |
| Balanced accuracy | 0.977706 |
| 肿瘤 F1 | 0.995413 |
| 肿瘤召回率 | 0.995413 |
| 特异度 | 0.960000 |

混淆矩阵为 `[[24, 1], [1, 217]]`，行是真实类别、列是预测类别，顺序均为正常／肿瘤。共判对 241 个样本，正常和肿瘤各误判 1 个。正常测试样本只有 25 个，每错一个会使特异度变化 0.04，因此不能仅凭本次结果推断临床有效性。

[逐样本预测](../results/rf_test_manual/predictions.csv)包含 `sample_id`、`true_label`、`predicted_label` 和 `tumor_probability`。已有[核验记录](../results/rf_supplement/test_prediction_verification.json)确认 243 个样本 ID 唯一，且由预测重算的指标及混淆矩阵与原结果一致。

## 5. 运行与输出

在项目根目录使用 [requirements-rf.txt](../requirements-rf.txt) 所列依赖。实际环境为 Python 3.13.9、NumPy 2.3.5、pandas 2.3.3、SciPy 1.16.3、scikit-learn 1.7.2、joblib 1.5.2、threadpoolctl 3.5.0。候选与折顺序执行，RF 默认 2 线程，数值库限制为 2 线程。

训练入口为 [src/random_forest.py](../src/random_forest.py)。以下复现示例使用新的输出目录和模型路径；若路径已存在，应再换名称，避免覆盖已有结果。

```powershell
python -m src.random_forest --help
python -m src.random_forest --output-dir results/rf_reproduce_v1 --model-path models/rf_reproduce_v1.joblib --jobs 2
python -m src.random_forest --supplement --output-dir results/rf_supplement_reproduce_v1 --jobs 2
```

第二条命令复现原参数搜索并保存新模型；第三条只运行四组补充参数分析，不保存或替换模型。历史训练记录保留原入口名称 `src/train_rf.py`，当前运行应使用新名称。

独立测试脚本为 [test_rf_manual.py](../test_rf_manual.py)，命令为 `python test_rf_manual.py`。它固定读取 `models/rf_pipeline.joblib` 和同版本的 `data/processed/tcga_brca_20261029_v2/test_data.npz`，输出到 `results/rf_test_manual/`。该脚本再次执行会覆盖原测试指标、混淆矩阵和预测表，仅在需要重新评价时使用。

| 位置 | 内容 |
|---|---|
| [results/rf/](../results/rf/) | 原参数搜索、内部折指标及预测、基因重要性和运行配置。 |
| [results/rf_supplement/](../results/rf_supplement/) | 补充分析、基础对照、完整冻结参数及核验记录。 |
| [results/rf_test_manual/](../results/rf_test_manual/) | 独立测试指标、混淆矩阵和逐样本预测。 |
| `models/rf_pipeline.joblib` | 包含 D 筛选器和 RF 的完整冻结 Pipeline，按 Git 忽略规则单独共享。 |

选中基因的 Ensembl ID 和 RF 重要性见[基因表](../results/rf/selected_genes_importance.csv)。重要性反映模型的分裂贡献，不等于因果关系。

## 6. 模型交接

- 模型文件：`models/rf_pipeline.joblib`，在全部 976 个训练样本上拟合。加载时需保留可导入的 `src.feature_selection` 及兼容依赖；输入保持原始 60,660 列及顺序，直接调用完整 Pipeline。
- 模型 SHA256：`f2a231ddc814803334770447590d622c378c76c5d35ad919d3ada8bf045a7b4d`，依据[原模型记录](../results/rf/best_model.json)；[保护核验记录](../results/rf_supplement/verification.json)显示补充分析前后未改变。
- 模型交付方式：从项目本地路径 `models/rf_pipeline.joblib` 单独交付模型文件，并附依赖记录与模型 SHA256；模型不纳入 Git 提交。

已确认 F 的 SVM 与 RF 使用相同的训练集、测试集划分。复现时沿用 C 的固定划分，不自行重新拆分；内部验证建议沿用 D 的患者五折，SVM 的特征筛选和标准化均在 Pipeline 内、每个训练折中拟合。

交给 G 的 RF 测试预测位于 `results/rf_test_manual/predictions.csv`。G 按 `sample_id` 对齐 RF、SVM 的预测，核对样本集合与真实标签，使用相同标签定义（0=正常、1=肿瘤）和指标进行比较。RF 提供肿瘤概率，SVM 提供肿瘤方向的决策得分，二者都可用于 ROC-AUC，但决策得分不能当作概率。补充分析与冻结模型的正式测试成绩分开报告。
