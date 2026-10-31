# GitHub 交付记录

- 交付日期：2026-10-31。
- 仓库：[Bioinformatics-Disease-Classification](https://github.com/yz-Yezi-Genm/Bioinformatics-Disease-Classification)。
- 关联任务：[issue #5](https://github.com/yz-Yezi-Genm/Bioinformatics-Disease-Classification/issues/5)。
- 交付分支：[codex/issue-5-feature-selection](https://github.com/yz-Yezi-Genm/Bioinformatics-Disease-Classification/tree/codex/issue-5-feature-selection)。
- 审核入口：[PR #6](https://github.com/yz-Yezi-Genm/Bioinformatics-Disease-Classification/pull/6)。
- 科学代码与结果的首次提交：[7cab77c7580438f85f9156ebaad03042b88ff5dd](https://github.com/yz-Yezi-Genm/Bioinformatics-Disease-Classification/commit/7cab77c7580438f85f9156ebaad03042b88ff5dd)。其后补充本记录和交付状态文档，最新提交以 PR 为准。

## 状态与范围

文件已按 src、docs、results/feature_selection、results/figures、scripts、examples 和 tests 的相对路径提交，包含独立 README_FEATURE_SELECTION.md 与 requirements-feature-selection.txt。保留团队原有 README、依赖、预处理代码和数据；未新增原始数据、虚拟环境、模型权重或交付 ZIP。

记录生成时 PR 已创建，等待组长审核，未合并 main，未关闭 issue #5、未发布 issue 评论或代勾验收项。E/F 正式接入和最终模型确认仍需协作。该状态为本次交付时的记录，后续以 GitHub 实际状态为准。

## 检查与本地包

8 项单元测试通过，真实数据结果复核为 PASS，包括训练输入哈希、名单列映射、患者分组隔离、30 次参考拟合的折外指标与汇总值。未加载保留测试数组。

本地压缩包包含这份提交记录、代码、方法报告、基因列表、分析图、复现与验证脚本及交接说明。包内 DELIVERY_SHA256.txt 校验各文件，压缩包旁的 .sha256 文件校验整个压缩包。包内 .gitignore 用于独立本地交付，未替换团队仓库的 .gitignore。
