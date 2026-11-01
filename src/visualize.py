"""Create separate Random Forest and SVM result pages and comparison figures.

Run from the project root: python -m src.visualize
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if __package__:
    from .evaluation_io import load_inputs, file_sha, COLORS, INK, MUTED, GRID, KEYS
else:
    sys.path.insert(0, str(ROOT))
    from src.evaluation_io import load_inputs, file_sha, COLORS, INK, MUTED, GRID, KEYS

MODELS = ["Random Forest", "SVM"]
FOLDERS = {"Random Forest": "random_forest", "SVM": "svm"}
PREFIXES = {"Random Forest": "rf", "SVM": "svm"}
LABELS = ["Accuracy", "Precision (tumor)", "Recall (tumor)", "F1-score (tumor)",
          "ROC-AUC", "Specificity (normal)", "Balanced accuracy"]
MODEL_CARDS = [
    ("confusion_matrix", "混淆矩阵", "展示本模型的 TN、FP、FN、TP，数量和真实类别内的比例同时标注。"),
    ("roc_curve", "ROC 曲线与 AUC", "仅显示本模型，左侧完整范围，右侧局部放大。横轴为假正例率，纵轴为真正例率。"),
    ("metrics", "评价指标与样本计数", "仅显示本模型的七项指标，以及正常、肿瘤样本数和混淆矩阵计数。"),
    ("misclassified_samples", "本模型误分类样本", "只列出本模型判错的样本，标出真实类别、预测类别、FP/FN 及对应分数。"),
]
COMPARISON_CARDS = [
    ("conditions", "对比依据与输入特征", "对比维度：同一测试集、类别分布、输入特征数量、模型设置与分数含义。两模型均使用 100 个筛选基因。"),
    ("metrics", "七项评价指标对比", "对比维度：Accuracy、Precision、Recall、F1-score、ROC-AUC、Specificity、Balanced accuracy。每行明确写出指标名称。"),
    ("confusion_matrix", "混淆矩阵计数与比例对比", "对比维度：TN、FP、FN、TP 及各真实类别内的比例。行是真实类别，列是预测类别。"),
    ("roc_curve", "ROC-AUC 与分数排序表现对比", "对比维度：ROC-AUC，以及不同阈值下 FPR 与 TPR 的关系；不直接比较两种原始分数的大小。"),
    ("misclassified_samples", "误分类样本对比", "对比维度：样本 ID、真实类别、预测类别和错误类型。概率与决策分数仅作为各模型的辅助信息。"),
]


class Renderer:
    def __init__(self, metrics, predictions, output, dpi):
        self.metrics, self.predictions, self.output, self.dpi = metrics, predictions, output, dpi
        self.figure_paths = []

    def save(self, figure, folder, name):
        dest = self.output / folder
        dest.mkdir(parents=True, exist_ok=True)
        for extension in ["png", "svg"]:
            target = dest / f"{name}.{extension}"
            temporary = dest / f".{name}.tmp.{extension}"
            try:
                figure.savefig(temporary, dpi=self.dpi)
                if extension == "png":
                    with Image.open(temporary) as saved_image:
                        saved_image.verify()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            self.figure_paths.append(target)
        plt.close(figure)

    def title(self, fig, title, subtitle, color=INK):
        fig.text(0.5, 0.94, title, ha="center", fontsize=21, fontweight="bold", color=color)
        fig.text(0.5, 0.875, subtitle, ha="center", fontsize=10.5, color=MUTED)

    def confusion(self, models, folder):
        pair = len(models) == 2
        fig = plt.figure(figsize=(11.6, 6.0) if pair else (6.6, 6.0))
        self.title(fig, "Confusion-matrix comparison" if pair else f"{models[0]}: confusion matrix",
                   "Compared: TN, FP, FN, TP and within-class percentages." if pair else "Normal = 0; tumor = 1. Positive class = tumor.")
        gs = fig.add_gridspec(1, len(models) + 1, width_ratios=[1] * len(models) + [0.04],
                             left=0.09 if pair else 0.21, right=0.90 if pair else 0.85,
                             bottom=0.20, top=0.76, wspace=0.25)
        cmap = LinearSegmentedColormap.from_list("teal", ["#F1F6F7", "#90C1C6", "#174F62"])
        for k, model in enumerate(models):
            ax = fig.add_subplot(gs[0, k]); row = self.metrics.loc[model]
            cm = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
            proportions = cm / cm.sum(axis=1, keepdims=True)
            im = ax.imshow(proportions, vmin=0, vmax=1, cmap=cmap)
            for i in range(2):
                for j in range(2):
                    color = "white" if proportions[i, j] > 0.6 else INK
                    ax.text(j, i - 0.26, [["TN", "FP"], ["FN", "TP"]][i][j], ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=29, fontweight="bold", color=color)
                    ax.text(j, i + 0.27, f"{proportions[i, j]:.2%}", ha="center", va="center", fontsize=11.5, color=color)
            if pair: ax.set_title(model, color=COLORS[model], fontweight="bold", pad=12)
            ax.set_xticks([0, 1], ["Normal (0)", "Tumor (1)"])
            ax.set_yticks([0, 1], ["Normal (0)", "Tumor (1)"])
            ax.set_xlabel("Predicted class", labelpad=10)
            ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
            ax.grid(which="minor", color="white", linewidth=3)
            ax.tick_params(which="both", length=0, pad=8)
            for spine in ax.spines.values(): spine.set_visible(False)
        bar = fig.colorbar(im, cax=fig.add_subplot(gs[0, -1]), ticks=[0, .25, .5, .75, 1])
        bar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        bar.ax.tick_params(length=0, labelsize=9); bar.outline.set_visible(False)
        fig.text(.022, .47, "True class", rotation=90, ha="center", va="center", fontsize=11)
        fig.text(.5, .065, "Cells show sample counts and percentages within each true class. Color scale: 0%-100%.", ha="center", fontsize=9, color=MUTED)
        self.save(fig, folder, "confusion_matrix")

    def roc(self, models, folder):
        pair = len(models) == 2
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.0))
        fig.subplots_adjust(left=.075, right=.96, bottom=.20, top=.72, wspace=.30)
        self.title(fig, "ROC-AUC comparison" if pair else f"{models[0]}: ROC curve",
                   "Compared: ROC-AUC and the TPR-FPR relationship across score thresholds." if pair else "Continuous model scores. Positive class = tumor (1).")
        handles = []
        for model in models:
            prefix = PREFIXES[model]
            fpr, tpr, _ = roc_curve(self.predictions.y_true, self.predictions[f"{prefix}_score"], drop_intermediate=False)
            auc = roc_auc_score(self.predictions.y_true, self.predictions[f"{prefix}_score"])
            color, marker = COLORS[model], "o" if prefix == "rf" else "D"
            style = "-" if prefix == "rf" else (0, (4, 2))
            handles.append(Line2D([0], [0], color=color, linestyle=style, lw=2.2, label=f"{model}: AUC = {auc:.6f}"))
            row = self.metrics.loc[model]
            for ax in axes:
                ax.plot(fpr, tpr, color=color, linestyle=style, lw=2.2)
                ax.scatter(row.fp / row.n_normal, row.recall, color=color, marker=marker, s=55, edgecolor="white", linewidth=1, zorder=5)
        axes[0].plot([0, 1], [0, 1], color="#A5AFB8", lw=1, linestyle=(0, (3, 3)))
        axes[0].set(xlim=(-.015, 1.015), ylim=(-.015, 1.025))
        axes[1].set(xlim=(-.003, .085), ylim=(.985, 1.0008))
        axes[1].set_xticks([0, .02, .04, .06, .08]); axes[1].set_yticks([.985, .990, .995, 1])
        axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        for ax, title in zip(axes, ["Full range", "Upper-left detail"]):
            ax.set_title(title, loc="left", fontweight="bold", pad=13)
            ax.set_xlabel("False positive rate (FPR)"); ax.set_ylabel("True positive rate (TPR)")
            ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=0, pad=8)
            ax.grid(color=GRID, linewidth=.8); ax.set_axisbelow(True)
        fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, .805), ncol=len(models), frameon=False, fontsize=11, columnspacing=2.5)
        fig.text(.5, .065, "Markers show the original model decisions. Right panel uses an enlarged coordinate range.", ha="center", fontsize=9.5, color=MUTED)
        self.save(fig, folder, "roc_curve")

    def table(self, folder, name, title, subtitle, headers, rows, widths, footer="", colors=None):
        height = max(4.8, 2.3 + len(rows) * .39)
        fig, ax = plt.subplots(figsize=(11.6 if len(headers) >= 3 else 8.0, height))
        fig.subplots_adjust(left=.04, right=.96, bottom=.16, top=.76)
        self.title(fig, title, subtitle)
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=headers, colWidths=widths, cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False); table.set_fontsize(10.5)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("white"); cell.set_linewidth(2)
            if r == 0:
                cell.set_facecolor(INK); cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#F0F4F6" if r % 2 else "white")
                cell.set_text_props(color=(colors[c] if colors else INK))
            if c == 0: cell.set_text_props(ha="left"); cell.PAD = .06
        if footer: fig.text(.5, .065, footer, ha="center", color=MUTED, fontsize=9)
        self.save(fig, folder, name)

    def model_metrics(self, model):
        counts = [("Test samples", "n_samples"), ("Normal samples", "n_normal"), ("Tumor samples", "n_tumor")]
        rows = [[label, str(int(self.metrics.loc[model, key]))] for label, key in counts]
        rows += [[label, f"{self.metrics.loc[model, key]:.6f}"] for label, key in zip(LABELS, KEYS)]
        rows += [[label, str(int(self.metrics.loc[model, key]))] for label, key in [("True negatives (TN)", "tn"), ("False positives (FP)", "fp"), ("False negatives (FN)", "fn"), ("True positives (TP)", "tp")]]
        self.table(FOLDERS[model], "metrics", f"{model}: evaluation metrics", "This model only. Positive class = tumor (1).", ["Metric / count", model], rows, [.60, .40], "Rates: six decimal places. Counts: integers.", [INK, COLORS[model]])

    def errors(self, models, folder):
        pair = len(models) == 2
        mask = np.zeros(len(self.predictions), dtype=bool)
        for model in models: mask |= self.predictions[f"{PREFIXES[model]}_y_pred"].ne(self.predictions.y_true)
        wrong = self.predictions.loc[mask].sort_values("sample_id")
        rows = []
        for _, row in wrong.iterrows():
            cells = [row.sample_id, "Tumor (1)" if row.y_true else "Normal (0)"]
            for model in models:
                prefix = PREFIXES[model]; label = int(row[f"{prefix}_y_pred"])
                status = "Correct" if label == row.y_true else ("FN" if row.y_true else "FP")
                term = "p(tumor)" if prefix == "rf" else "score"
                cells.append(f"{'Tumor' if label else 'Normal'} ({status})\n{term} = {row[f'{prefix}_score']:.6f}")
            rows.append(cells)
        if not rows: rows = [["No errors", "-"] + ["-"] * len(models)]
        self.table(folder, "misclassified_samples", "Misclassification comparison" if pair else f"{models[0]}: misclassified samples",
                   "Compared: sample ID, true label, predicted label and error type." if pair else f"Only errors from this model ({len(wrong)} samples). FP = false positive; FN = false negative.",
                   ["Sample ID", "True class", *models], rows, [.30, .15, .275, .275] if pair else [.39, .21, .40],
                   "RF probability and SVM decision score use different scales; do not compare their magnitudes." if pair else ("p(tumor) is the predicted tumor probability." if models[0] == "Random Forest" else "The decision score is not a probability; negative scores favor normal tissue."))

    def metric_comparison(self):
        fig = plt.figure(figsize=(12.1, 6.6))
        self.title(fig, "Classification-metric comparison", "Compared: Accuracy, Precision, Recall, F1-score, ROC-AUC, Specificity and Balanced accuracy.")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1], left=.225, right=.975, bottom=.18, top=.76, wspace=.12)
        ax, values = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]); y = np.arange(7)
        a = self.metrics.loc["Random Forest", KEYS].to_numpy(dtype=float)
        b = self.metrics.loc["SVM", KEYS].to_numpy(dtype=float)
        for panel in [ax, values]:
            panel.set_ylim(6.6, -.9)
            for r in [0, 2, 4, 6]: panel.axhspan(r - .43, r + .43, color="#F3F6F8", zorder=0)
        for r, av, bv in zip(y, a, b): ax.plot([av, bv], [r - .1, r + .1], color="#B8C3CB", lw=1.2)
        ax.scatter(a, y - .1, color=COLORS[MODELS[0]], marker="o", s=55, edgecolor="white", zorder=4)
        ax.scatter(b, y + .1, color=COLORS[MODELS[1]], marker="D", s=45, edgecolor="white", zorder=4)
        ax.set_yticks(y, LABELS, fontsize=10.5); ax.set_xlim(.95, 1.003)
        ax.set_xticks([.95, .96, .97, .98, .99, 1]); ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlabel("Metric value (zoomed to 0.95-1.00)", labelpad=12)
        ax.spines[["top", "left", "right"]].set_visible(False); ax.tick_params(length=0, pad=10)
        ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
        values.set_xlim(0, 1); values.axis("off")
        for x, model in [(.27, MODELS[0]), (.80, MODELS[1])]: values.text(x, -.65, model, ha="center", color=COLORS[model], fontsize=10, fontweight="bold")
        for r, av, bv in zip(y, a, b):
            values.text(.27, r, f"{av:.6f}", va="center", ha="center", color=COLORS[MODELS[0]])
            values.text(.80, r, f"{bv:.6f}", va="center", ha="center", color=COLORS[MODELS[1]])
        fig.text(.5, .06, "Tumor is positive (1). Specificity is normal-class recall. Point colors match the named columns.", ha="center", fontsize=9.5, color=MUTED)
        self.save(fig, "comparison", "metrics")


def comparison_conditions(results, metrics):
    rf = json.loads((results / "rf_supplement/frozen_model_parameters.json").read_text())
    svm = json.loads((results / "svm/selected_parameters.json").read_text())
    a, b = metrics.loc[MODELS[0]], metrics.loc[MODELS[1]]
    return [
        ["Test samples", str(int(a.n_samples)), str(int(b.n_samples))],
        ["Normal / tumor samples", f"{int(a.n_normal)} / {int(a.n_tumor)}", f"{int(b.n_normal)} / {int(b.n_tumor)}"],
        ["Positive class", "Tumor (1)", "Tumor (1)"],
        ["Selected input features (genes)", str(rf["selector"]["k"]), str(svm["feature_count"])],
        ["Model setting", f"{rf['classifier']['n_estimators']} trees", f"{svm['parameters']['classifier__kernel'].upper()} kernel"],
        ["Model parameters", f"max_features = {rf['classifier']['max_features']}", f"C = {svm['parameters']['classifier__C']}; gamma = {svm['parameters']['classifier__gamma']}"],
        ["Score used for ROC-AUC", "Tumor probability", "Decision-function score"],
        ["Metrics compared", "Accuracy, Precision, Recall, F1", "Accuracy, Precision, Recall, F1"],
        ["Additional metrics compared", "ROC-AUC, Specificity, Balanced accuracy", "ROC-AUC, Specificity, Balanced accuracy"],
        ["Error outcomes compared", "TN, FP, FN, TP", "TN, FP, FN, TP"],
    ]


def write_pages(out):
    style = '''<style>body{margin:0;background:#f3f6f8;color:#172c3d;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:30px 22px 60px}h1{margin:10px 0;font-size:30px}h2{font-size:22px;margin:0}p{color:#62727d}section{background:white;padding:24px;border:1px solid #e0e7ec;border-radius:12px;margin:24px 0}img{max-width:100%;height:auto;display:block;margin:auto}nav{display:flex;gap:16px;flex-wrap:wrap}a{color:#167d8d;text-decoration:none}.downloads{display:flex;gap:20px;margin-top:12px}.dimension{color:#172c3d;background:#f0f5f7;padding:10px 14px;border-radius:6px}</style>'''
    nav = '<nav><a href="index.html">总入口</a><a href="random_forest.html">随机森林</a><a href="svm.html">SVM</a><a href="comparison.html">两模型对比</a></nav>'
    def page(title, content):
        return f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>{style}<main>{nav}<h1>{title}</h1>{content}</main></html>'
    home = '<p>结果分成三个独立页面。单模型页面只展示该模型；对比页面在每张图前明确写出对比维度。</p>'
    for href, title, desc in [("random_forest.html", "随机森林算法", "本算法的混淆矩阵、ROC、指标及误分类样本。"), ("svm.html", "SVM 算法", "本算法的混淆矩阵、ROC、指标及误分类样本。"), ("comparison.html", "两算法对比", "明确比较七项指标、混淆矩阵计数、ROC-AUC、错误样本及输入特征数量。")]:
        home += f'<section><h2><a href="{href}">{title}</a></h2><p>{desc}</p></section>'
    (out / "index.html").write_text(page("模型评估结果", home), encoding="utf-8")
    for filename, folder, title, cards in [("random_forest.html", "random_forest", "随机森林算法结果", MODEL_CARDS), ("svm.html", "svm", "SVM 算法结果", MODEL_CARDS), ("comparison.html", "comparison", "随机森林与 SVM 对比", COMPARISON_CARDS)]:
        content = '<p>同一批 243 个测试样本，肿瘤为正类（1），正常组织为负类（0）。单模型结果与对比结果分别展示。</p>'
        for name, heading, caption in cards:
            path = f"{folder}/{name}"
            content += f'<section><h2>{heading}</h2><p class="dimension">{caption}</p><a href="{path}.png" target="_blank"><img src="{path}.png" alt="{heading}"></a><div class="downloads"><a href="{path}.png" download>下载 PNG</a><a href="{path}.svg" download>下载 SVG</a></div></section>'
        (out / filename).write_text(page(title, content), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/model_views")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if not 72 <= args.dpi <= 600: raise ValueError("dpi must be between 72 and 600")
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise ValueError("Output directory must be new or empty")
    test = ROOT / "data/processed/tcga_brca_20261029_v2/test_data.npz"
    metrics, pred = load_inputs(args.results_dir, test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(metrics, pred, args.output_dir, args.dpi)
    for model in MODELS:
        folder = FOLDERS[model]
        renderer.confusion([model], folder)
        renderer.roc([model], folder)
        renderer.model_metrics(model)
        renderer.errors([model], folder)
    renderer.table("comparison", "conditions", "Comparison conditions and input features", "Explicit basis: same test samples, selected genes, model settings and score definitions.",
                   ["Comparison dimension", *MODELS], comparison_conditions(args.results_dir, metrics), [.34, .33, .33],
                   "A gene is an input feature. Accuracy and AUC are evaluation metrics; these are different concepts.")
    renderer.metric_comparison()
    renderer.confusion(MODELS, "comparison")
    renderer.roc(MODELS, "comparison")
    renderer.errors(MODELS, "comparison")
    write_pages(args.output_dir)
    audit = {"single_model_figures": {"Random Forest": 4, "SVM": 4}, "comparison_figures": 5,
             "comparison_dimensions": [*LABELS, "TN", "FP", "FN", "TP", "misclassified samples", "selected input gene count", "model settings"],
             "n_test_samples": len(pred), "positive_class": "tumor (1)", "png_dpi": args.dpi,
             "inputs_verified": True, "training_or_threshold_tuning": False,
             "script_sha256": file_sha(Path(__file__)), "created_utc": datetime.now(timezone.utc).isoformat(),
             "inputs": {str(p.relative_to(ROOT)): file_sha(p) for p in [args.results_dir / "metrics.csv", args.results_dir / "evaluation/aligned_predictions.csv", args.results_dir / "rf_supplement/frozen_model_parameters.json", args.results_dir / "svm/selected_parameters.json"]},
             "outputs": {str(p.relative_to(args.output_dir)): file_sha(p) for p in sorted(args.output_dir.rglob('*')) if p.is_file() and not p.name.startswith('.')}}
    (args.output_dir / "display_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved 8 single-model figures, 5 labelled comparisons and 4 HTML pages to {args.output_dir.resolve()}")


if __name__ == "__main__":
    try: main()
    except (ValueError, KeyError, OSError) as exc:
        print(f"Visualization failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
