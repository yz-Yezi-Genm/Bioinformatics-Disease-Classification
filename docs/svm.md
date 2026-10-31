# SVM 模型实验说明

对应 Issue #8：基于 RNA-seq 数据的 SVM 分类模型构建。本文说明算法原理、参数设置、实验过程及运行方式，并对应任务要求的五项交付成果。文件路径均相对于仓库根目录。

### 五项交付成果对应位置

| 要求 | 文件位置 | 文件说明 |
|---|---|---|
| 1. SVM 模型训练代码 | [src/svm_model.py](../src/svm_model.py) | 读取数据、训练模型、比较参数并生成预测的代码。 |
| 2. 算法原理、参数设置及实验过程说明 | 本文 `docs/svm.md` 第二节 | 说明模型如何分类、尝试了哪些参数，以及实际训练步骤。 |
| 3. 不同参数设置下的实验记录 | [results/svm/cv_results.csv](../results/svm/cv_results.csv) | 12 组参数的完整五折验证记录；本文第三节提供简表。 |
| 4. 测试集预测结果 | [results/svm/test_predictions.csv](../results/svm/test_predictions.csv) | 每个测试样本的 ID、真实标签、预测标签和预测得分。 |
| 5. 最终模型参数和运行说明 | [results/svm/selected_parameters.json](../results/svm/selected_parameters.json) 及本文第五节 | JSON 保存最终参数；第五节说明如何运行代码和加载模型。 |

## 一、模型训练代码

文件：`src/svm_model.py`。

代码完成数据读取、特征选择、标准化、训练集内部交叉验证调参、最终模型拟合以及测试集预测。

### 输入数据

使用 C 提供的 `data/processed/tcga_brca_20261029_v2/`，与 E 的随机森林实验保持一致。

| 输入 | 内容 |
|---|---|
| `train_data.npz` | 976 个训练样本，其中正常 88 个、肿瘤 888 个 |
| `test_data.npz` | 243 个独立测试样本，其中正常 25 个、肿瘤 218 个 |
| `feature_ids.tsv` | 60,660 个基因的原始列顺序 |
| `sample_manifest.tsv` | 样本 ID、患者 ID、标签与训练/测试归属 |
| `audit.json` | C 的数据校验记录，用于确认输入版本 |

标签为 0=正常、1=肿瘤。NPZ 内包含表达矩阵 `X`、标签 `y` 和样本 ID `sample_id`。保持原划分，训练与测试患者无重叠，不重新随机拆分。

## 二、算法原理、参数设置及实验过程

### 算法原理

支持向量机（SVM）通过寻找分类间隔较大的决策边界区分两类样本，并允许一定训练误差。支持向量是对该边界起关键作用的训练样本。本实验采用 scikit-learn 的 `SVC`，比较线性核和径向基函数核（RBF）。

- `C`：控制对训练误差的惩罚。较小的 C 对应更强的正则化，较大的 C 更强调减少训练误差。
- `kernel`：`linear` 构建线性边界；`rbf` 可以描述非线性关系。
- `gamma`：RBF 核参数，较大时单个样本的影响更局部，较小时影响范围更广。该参数对 linear 核不起作用。

SVM 对特征尺度敏感，因此使用 `StandardScaler` 将选出的基因表达特征标准化。类别比例不平衡，固定使用 `class_weight='balanced'`，由每次拟合的数 据计算类别权重。

### 参数设置

与 E 统一使用 D 的 `create_selector(k=100)`。k 固定为 100，本次只调整 SVM 的核函数、C 和 gamma。

| 核函数 | C 候选值 | gamma 候选值 | 组合数 |
|---|---|---|---:|
| linear | 0.1、1、10 | 不适用 | 3 |
| rbf | 0.1、1、10 | scale、0.001、0.01 | 9 |

这里gamma不适用的意思是gamma的值对模型没有影响，类似于没有这个参数。

共 12 组参数。RBF 的 `gamma='scale'` 根据拟合数据的特征数量和方差自动计算；在标准化后的 100 个非恒定特征上通常约为 0.01，因此与显式设置 0.01 的结果可能一致。

### 实验过程

**第 1 步：准备数据，先保留测试集。** 使用 C 已经分好的 976 个训练样本和 243 个测试样本，检查文件版本、样本标签和基因列顺序。前面的训练与调参只使用 976 个训练样本，243 个测试样本留到最后检验效果。

**第 2 步：在训练集内部轮流训练和验证。** 沿用 D 已保存的五组样本分配（`results/feature_selection/cv_fold_assignments.csv`）。每次取其中四组训练，剩下一组验证，轮换五次，让每组都有一次作为验证数据的机会。同一患者的样本始终在同一组。这里只是在训练集内部做比较，没有改变 C 的训练集和测试集划分。

**第 3 步：每次训练都先选基因，再标准化，最后训练 SVM。** 先让 D 的筛选程序根据当前四组训练样本，从 60,660 个基因中选出 100 个；再根据这四组样本计算标准化所需的均值和标准差，减少不同基因数值尺度的影响；最后把处理后的表达数据和已知的正常／肿瘤标签交给 SVM，让它学习区分两类样本的规则。代码把这三个步骤串在一个 Pipeline 中，保证每次训练都按相同顺序执行。

**第 4 步：用剩下的一组检查本次训练效果。** 验证样本只沿用刚才选出的基因和标准化规则，不重新选基因，也不参与模型训练。模型先给出预测，再与真实标签比较，计算 AUC、准确率、F1 等指标。完成五轮后，计算这组参数的平均验证成绩。

**第 5 步：比较不同参数并确定最终设置。** 用上述相同方法比较 12 组参数，每组训练和验证五次，共进行 60 次训练。这些工作由 GridSearchCV 自动完成。按事先确定的规则，优先选择平均验证 AUC 最高的参数；完全并列时比较平均 balanced accuracy，再并列则按候选顺序选择。本次选中 RBF 核、C=0.1、gamma=0.001，并保存全部参数的实验记录。

**第 6 步：用全部训练样本训练最终模型。** 参数确定后，重新使用完整的 976 个训练样本选择 100 个基因、计算标准化规则并训练 SVM。这次得到的是最终模型，而不是直接拿前面某一折的临时模型使用。最终模型连同基因选择器和标准化器一起保存，方便以后直接预测。

**第 7 步：检验测试集并导出预测结果。** 最后才让最终模型预测预留的 243 个测试样本。测试样本仍只使用训练阶段得到的基因选择和标准化规则。保存每个样本的 ID、真实标签、预测标签和决策得分，交给 G 统一评估。不能根据这次测试成绩再修改参数、基因数量或判定阈值。

使用 `probability=False`，输出 `decision_function` 得分。得分越大越倾向肿瘤类，采用默认零决策边界。该分数不是概率。

## 三、不同参数设置下的实验记录

文件：`results/svm/cv_results.csv`。

每行对应一组参数，包含参数值、五折各项指标、均值、标准差和耗时。`split0` 至 `split4` 分别对应 D 的第 1 至第 5 折。标准差采用 GridSearchCV 的总体标准差口径（ddof=0）。

可以用 Excel 打开该 CSV。阅读时优先查看以下列：

| 列名 | 通俗说明 |
|---|---|
| `param_classifier__kernel`、`param_classifier__C`、`param_classifier__gamma` | 本行实验使用的核函数、C 和 gamma；linear 行的 gamma 为空，因为不适用。 |
| `split0_test_roc_auc` 至 `split4_test_roc_auc` | 这组参数在五次验证中的 AUC。 |
| `mean_test_roc_auc` | 五次验证的平均 AUC，本次选参数主要依据这一列。 |
| `std_test_roc_auc` | 五次验证 AUC 的波动大小。 |
| `mean_test_accuracy`、`mean_test_balanced_accuracy`、`mean_test_tumor_f1` | 平均准确率、平衡准确率和肿瘤类 F1。 |
| `mean_fit_time` | 每一折完成筛选、标准化和 SVM 训练的平均耗时，单位为秒。 |

注意：这些列名中的 `test` 是 sklearn 对交叉验证评分的命名，在此指每轮留出的验证组，**不是最后的 243 个独立测试样本**。

本次五折平均成绩如下：

| 核函数 | C | gamma | ROC-AUC | Balanced accuracy | 肿瘤 F1 |
|---|---:|---|---:|---:|---:|
| linear | 0.1 | 不适用 | 0.999614 | 0.980994 | 0.996626 |
| linear | 1 | 不适用 | 0.999151 | 0.980433 | 0.996059 |
| linear | 10 | 不适用 | 0.999151 | 0.980433 | 0.996059 |
| rbf | 0.1 | scale | 0.999240 | 0.986005 | 0.990899 |
| rbf | 0.1 | 0.001 | 0.999621 | 0.984313 | 0.989173 |
| rbf | 0.1 | 0.01 | 0.999240 | 0.986005 | 0.990899 |
| rbf | 1 | scale | 0.999240 | 0.990744 | 0.996050 |
| rbf | 1 | 0.001 | 0.999563 | 0.994379 | 0.994331 |
| rbf | 1 | 0.01 | 0.999240 | 0.990744 | 0.996050 |
| rbf | 10 | scale | 0.999236 | 0.986550 | 0.997186 |
| rbf | 10 | 0.001 | 0.999497 | 0.997753 | 0.997743 |
| rbf | 10 | 0.01 | 0.999236 | 0.986550 | 0.997186 |

最终参数按预定 AUC 规则选择，并非所有指标均最佳。最佳 RBF 与最佳 linear 的验证 AUC 仅相差约 0.000007，不能据此宣称 RBF 显著优于线性核。交叉验证用于选型，不能把最佳验证成绩当作独立测试成绩。

## 四、测试集预测结果

文件：`results/svm/test_predictions.csv`，共 243 行样本记录，交给 G 进行统一评估。

| 字段 | 含义 |
|---|---|
| `sample_id` | 样本 ID，用于与 E 的预测结果对齐 |
| `y_true` | 真实标签，0=正常、1=肿瘤 |
| `y_pred` | 预测标签，0=正常、1=肿瘤 |
| `decision_score` | 肿瘤方向的连续决策分数，用于 ROC/AUC，不是概率 |

根据该表计算的本次测试结果：

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.991770（99.18%） |
| ROC-AUC | 0.999083 |
| Balanced accuracy | 0.995413 |
| 肿瘤 Precision | 1.000000 |
| 肿瘤 Recall | 0.990826 |
| 正常 Recall／特异度 | 1.000000 |
| 肿瘤 F1 | 0.995392 |

共判对 241/243 个样本：25 个正常样本全部判对，218 个肿瘤样本中 216 个判对、2 个误判为正常。测试正常样本仅 25 个，结果只描述本次 TCGA 组织分类划分，不能视为临床筛查效果。最终随机森林与 SVM 的统一对比由 G 完成。

## 五、最终模型参数和运行说明

最终参数文件：`results/svm/selected_parameters.json`。

| 项目 | 最终设置 |
|---|---|
| 特征选择 | D 的 create_selector，k=100 |
| 标准化 | StandardScaler，训练流程内拟合 |
| kernel | rbf |
| C | 0.1 |
| gamma | 0.001 |
| class_weight | balanced |
| probability | False，使用决策分数 |

最终配置的平均验证 AUC 为 0.999621，五折总体标准差为 0.000233。

### 环境与执行命令

实测 Python 3.13.3、scikit-learn 1.8.0。`requirements-svm.txt` 是运行代码的依赖文件。以下命令均在仓库根目录执行：

```powershell
python -m pip install -r requirements-svm.txt
python -m src.svm_model
```

也支持 `python src/svm_model.py`。程序默认输出到 `results/svm/`，完整模型保存在本地 `models/svm_pipeline.joblib`。模型由 Git 忽略，不直接上传 GitHub。

已有结果和模型不会被覆盖。复现本次实验时指定新路径：

```powershell
python -m src.svm_model --output-dir results/svm_reproduction --model-file models/svm_reproduction.joblib
```

参数搜索范围位于 `src/svm_model.py` 的 `--c-values`、`--gamma-values` 默认值及 `parameter_grid()` 中，也可通过命令行指定：

```powershell
python -m src.svm_model --c-values 0.1 1 10 --gamma-values scale 0.001 0.01 --output-dir results/svm_reproduction --model-file models/svm_reproduction.joblib
```

上述为本次预设搜索范围的复现方式。独立测试结果已生成，不应根据测试成绩继续调参挑选模型。

### 加载模型

在仓库根目录、相同依赖环境下执行，保留 D 的 `src/feature_selection.py`：

```python
import joblib
import numpy as np

model = joblib.load('models/svm_pipeline.joblib')
with np.load('data/processed/tcga_brca_20261029_v2/test_data.npz', allow_pickle=False) as test:
    prediction = model.predict(test['X'])
    score = model.decision_function(test['X'])
```

输入保持原始 60,660 列及同一基因顺序。完整 Pipeline 已包含筛选和标准化，无需提前手动裁剪为 100 列。

算法依据：[SVC 官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)、[GridSearchCV 官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)、[防止数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html)。
