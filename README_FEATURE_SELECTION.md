# D：TCGA-BRCA 特征选择交付包

对应 [issue #5：RNA-seq 基因表达数据的特征筛选与分析](https://github.com/yz-Yezi-Genm/Bioinformatics-Disease-Classification/issues/5)。交付日期：2026-10-31。

本包已完成真实训练数据上的筛选代码、基因列表、统计、四张图、方法报告和 E/F 接入示例，并补充了训练集内部的 RF/SVM 参考实验。使用 C 的 `TCGA-BRCA-experiment1-preprocessed-20261029-v2.zip`；不包含大型输入数据或 Python 虚拟环境。

## 先看什么

- `docs/交付说明.md`：要交哪些文件、哪些事项仍需 E/F/G 确认。
- `docs/feature_selection.md`：完整方法、真实结果、可复现命令、局限。
- `docs/汇报说明.md`：供小组 PPT／研究报告整合的简短说明。
- `examples/pipeline_integration.py`：E/F 如何正确使用模块。

## 主要结果

976 个训练样本、877 位患者、60,660 个基因。剔除 2,648 个训练集恒定特征后，剩余 58,012 个；分别筛选 100、500、1000 个基因。

在固定模型设置和患者分组的五折交叉验证中，按预先约定的 ROC-AUC 规则，RF 与 SVM 均推荐 k=100 作为参考设置。RF 的平均验证 AUC 为 0.999844，SVM 为 0.999240。这些是训练集内部参考分数，不是独立测试集结果，也不是 E/F 已确认的最终模型结果。

100 基因方案中，77 个基因在五个训练折均被选择；详细记录见 `selection_stability.csv`。

## 复现（在包的根目录执行）

以下命令以 Windows PowerShell、Python 3.12 为例；先保证 `python` 指向所需版本。原工作区已有 `.venv` 可直接使用。

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-feature-selection.txt
& .\.venv\Scripts\python.exe scripts/prepare_data.py --archive "C:\Users\丁鑫\Downloads\TCGA-BRCA-experiment1-preprocessed-20261029-v2.zip"
& .\.venv\Scripts\python.exe -m src.feature_selection --data-dir data/processed/tcga_brca_20261029_v2
& .\.venv\Scripts\python.exe -m scripts.run_reference_cv --data-dir data/processed/tcga_brca_20261029_v2 --jobs 2
& .\.venv\Scripts\python.exe -m scripts.make_figures
& .\.venv\Scripts\python.exe -m scripts.verify_results --data-dir data/processed/tcga_brca_20261029_v2
```

在其他电脑上修改 `--archive` 为实际路径。依赖锁定在 `requirements-feature-selection.txt`。输入压缩包 SHA256 为 `7cb784fe1ea37270599feaabe65f9634c7a277b8f16812d610e5bb597ba75ba4`。

## 交付边界

- 已完成 D 可独立交付内容，以及固定模型下真实可复现的 k 对比参考实验。
- 尚未声称 E/F 已采用该设置；最终训练、调参和保留测试集评价由 E/F/G 完成。
- 特征选择与参考实验没有加载 `test_data.npz`；加载器只使用训练数组及样本清单做对齐／患者隔离检查。
- 上传到团队仓库时，按本包的相对路径合入，保留团队原有 README 与 requirements。不要上传 `data/`、`.venv/` 或大型中间文件。
- 尚未向 GitHub 推送、评论或关闭 issue。本包是可审阅的本地交付成果。
