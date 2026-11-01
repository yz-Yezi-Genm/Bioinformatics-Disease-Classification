# SVM 复现结果核对使用说明

本包只增加 7 个文件，用于核对项目已有的 SVM 复现结果。包内不包含
`selected_parameters.json`、`test_metrics.json` 或 `test_predictions.csv` 的替换版本。

## 复制到项目

1. 将本压缩包解压到一个单独的文件夹。
2. 打开里面的 `Bioinformatics-Disease-Classification-main` 文件夹。
3. 将里面的 README 文件，以及 `docs`、`src`、`tests`、`results` 文件夹复制到现有项目根目录，合并同名文件夹。
4. 如果提示有同名文件，选择“跳过这些文件”，保留现有版本，不覆盖。

项目原有的 `results/svm/` 和已有的 `results/svm_reproduce_v1/` 都保留。
不需要再次合并此前的复现压缩包；此前复制进去的文件可以保留。
本次结论以 `docs/svm_reproduce_v1_verification.md` 和对应的 `summary.json` 为准。
其他运行产生的预测，应根据其文件摘要单独判断。

## 先查看已有核对结果

打开 `docs/svm_reproduce_v1_verification.md`。

这次使用项目已有的三个复现文件进行核对：243 个样本的 ID、真实标签、预测标签和决策得分全部完全一致；
两份预测 CSV 的 SHA-256 也完全相同。参数字段相同，复现测试指标与预测重新计算的指标一致。
因此原有随机森林、SVM 及对比图表继续使用，无需重画。

本记录核实的是交付文件及指标，并不单凭这三个文件认定整个训练过程已得到独立复现。

## 在自己的电脑上重新核对

在项目根目录空白处右键，选择“在终端中打开”。安装评估依赖：

```powershell
py -m pip install -r requirements-evaluation.txt
```

使用新的输出名称运行，保留包内核对记录：

```powershell
py -m src.check_svm_prediction_files --output-dir results/svm_reproduce_v1_local_check --report docs/svm_reproduce_v1_local_check.md
```

运行成功后打开 `docs/svm_reproduce_v1_local_check.md`。再次运行时换一个未使用的输出目录和报告文件名。
如果终端不识别 `py`，将两条命令中的 `py` 换成 `python`。

程序读取的复现文件须位于：

- `results/svm_reproduce_v1/selected_parameters.json`
- `results/svm_reproduce_v1/test_predictions.csv`
- `results/svm_reproduce_v1/test_metrics.json`

程序拒绝覆盖已有输出。`exact_match` 表示预测标签和决策得分逐值相等；
`within_tolerance` 表示标签一致、得分仅在绝对容差 `1e-12` 内一致，不能称为完全相同；
`different` 表示需要核实差异后，再用复现结果另存评价图表。

## 本包新增文件

- `README_SVM_RESULT_CHECK.md`：使用说明。
- `docs/svm_reproduce_v1_verification.md`：本次核对记录。
- `src/check_svm_prediction_files.py`：核对程序。
- `tests/test_svm_prediction_file_check.py`：样本对齐、差异识别、指标验证和输出保护测试。
- `results/svm_reproduce_v1_verification/summary.json`：结论、输入摘要和运行环境。
- `results/svm_reproduce_v1_verification/sample_comparison.csv`：逐样本比较。
- `results/svm_reproduce_v1_verification/metrics_comparison.csv`：指标对照。

上传 GitHub 时按上述相对路径添加文件，保留原有目录结构。
