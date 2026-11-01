"""Verify saved SVM reproduction files without retraining or replacing outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .evaluate import (ROOT, DATA_DIR, METRICS, binary_values, compute_metrics,
                       load_reference, validate_ids, validate_manifest)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_predictions(path):
    table = pd.read_csv(path, dtype={"sample_id": str}, encoding="utf-8-sig", float_precision="round_trip")
    required = ["sample_id", "y_true", "y_pred", "decision_score"]
    if not set(required).issubset(table):
        raise ValueError(f"Missing SVM prediction columns: {path}")
    table = table[required].rename(columns={"decision_score": "score"})
    validate_ids(table, str(path))
    for key in ["y_true", "y_pred"]:
        table[key] = binary_values(table[key], key)
    table["score"] = pd.to_numeric(table.score, errors="raise")
    if not np.isfinite(table.score).all():
        raise ValueError("Decision scores must be finite")
    return table


def compare_tables(original, reproduced, reference, atol=1e-12):
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("Absolute tolerance must be finite and nonnegative")
    for frame, name in [(original, "original"), (reproduced, "reproduced"), (reference, "reference")]:
        validate_ids(frame, name)
    expected = set(reference.sample_id)
    for frame, name in [(original, "original"), (reproduced, "reproduced")]:
        if set(frame.sample_id) != expected:
            raise ValueError(f"{name}: sample IDs do not match the frozen test set")
    a, b = [f.set_index("sample_id").loc[reference.sample_id].reset_index() for f in [original, reproduced]]
    for frame in [a, b]:
        for key in ["y_true", "y_pred"]:
            frame[key] = binary_values(frame[key], key)
        if not np.array_equal(frame.y_true, reference.y_true):
            raise ValueError("True labels do not match the frozen test set")
        frame["score"] = pd.to_numeric(frame.score, errors="raise")
        if not np.isfinite(frame.score).all():
            raise ValueError("Decision scores must be finite")
    delta = np.abs(a.score.to_numpy() - b.score.to_numpy())
    detail = pd.DataFrame({
        "sample_id": reference.sample_id.to_numpy(),
        "original_y_true": a.y_true, "reproduced_y_true": b.y_true,
        "original_y_pred": a.y_pred, "reproduced_y_pred": b.y_pred,
        "original_decision_score": a.score, "reproduced_decision_score": b.score,
        "true_label_equal": a.y_true.eq(b.y_true), "prediction_equal": a.y_pred.eq(b.y_pred),
        "score_exact_equal": a.score.eq(b.score), "absolute_score_difference": delta,
        "score_within_tolerance": delta <= atol})
    summary = {
        "n_test_samples": len(reference), "sample_ids_equal": True,
        "true_labels_equal": bool(detail.true_label_equal.all()),
        "predicted_labels_equal": bool(detail.prediction_equal.all()),
        "decision_scores_exact_equal": bool(detail.score_exact_equal.all()),
        "decision_scores_within_tolerance": bool(detail.score_within_tolerance.all()),
        "predicted_label_mismatch_count": int((~detail.prediction_equal).sum()),
        "exact_score_difference_count": int((~detail.score_exact_equal).sum()),
        "max_absolute_score_difference": float(delta.max()),
        "absolute_tolerance": atol, "relative_tolerance": 0,
        "row_order_equal": original.sample_id.tolist() == reproduced.sample_id.tolist()}
    summary["prediction_consistency"] = ("exact_match" if summary["predicted_labels_equal"]
        and summary["decision_scores_exact_equal"] else "within_tolerance" if summary["predicted_labels_equal"]
        and summary["decision_scores_within_tolerance"] else "different")
    return summary, detail, a, b


def validate_reported_metrics(reported, predictions, atol=1e-12):
    values = compute_metrics(predictions, "SVM")
    mapping = {"n_samples": "n_samples", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
               "roc_auc": "roc_auc", "tumor_precision": "precision", "tumor_recall": "recall",
               "specificity": "specificity", "tumor_f1": "f1_score"}
    for stored, computed in mapping.items():
        actual = reported[stored]
        if not isinstance(actual, (int, float)) or not np.isfinite(actual) or abs(actual - values[computed]) > atol:
            raise ValueError(f"test_metrics.json disagrees with predictions: {stored}")
    macro = reported["macro_f1"]
    if not isinstance(macro, (int, float)) or not np.isfinite(macro) or abs(macro - f1_score(predictions.y_true, predictions.y_pred, average="macro")) > atol:
        raise ValueError("test_metrics.json disagrees with predictions: macro_f1")
    expected = [[values['tn'], values['fp']], [values['fn'], values['tp']]]
    if reported["confusion_matrix"] != expected:
        raise ValueError("test_metrics.json disagrees with predictions: confusion_matrix")
    if reported["confusion_matrix_order"] != "rows=true, columns=predicted; labels=[0,1]":
        raise ValueError("Unexpected confusion-matrix order")
    return values


def path_label(path):
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def write_json(path, data):
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_report(path, summary, metrics):
    s = summary
    lines = ["# SVM 复现结果核对记录", "",
        "本记录补充原 `docs/evaluation.md`，核对项目已有的 `results/svm_reproduce_v1/` 三个交付文件。"
        "本次只读取和核对，不重新训练、不改参数、不改分类阈值，也不覆盖原文件。", "",
        "## 核对文件", "",
        "原预测为 `results/svm/test_predictions.csv`，复现预测为 `results/svm_reproduce_v1/test_predictions.csv`。"
        "同时读取复现参数和测试指标 JSON，核实参数字段及指标是否与预测表相符。"
        "测试 NPZ、样本清单和训练 NPZ 用来核对样本、真实标签及患者隔离。", "",
        "## 核对结果", "", "| 检查内容 | 结果 |", "|---|---|",
        f"| 样本 ID | {s['n_test_samples']} 个，均唯一，与参考测试集一致 |",
        "| 真实标签 | 与参考测试集及样本清单一致 |",
        f"| 预测标签 | 不一致 {s['predicted_label_mismatch_count']} 个 |",
        f"| 决策得分逐值比较 | 不一致 {s['exact_score_difference_count']} 个 |",
        f"| 决策得分最大绝对差 | {s['max_absolute_score_difference']:.17g} |",
        f"| 预测 CSV 的 SHA-256 | {'完全相同' if s['prediction_files_byte_identical'] else '不同，数值结论以逐样本比较为准'} |",
        "| 复现测试指标 JSON | 与预测表重新计算的指标一致 |",
        f"| 参数与原参数记录 | {'一致' if s['selected_parameters_equal'] else '不同'}（不比较运行时间戳） |",
        f"| 训练／测试患者重叠 | {s['split']['train_test_patient_overlap']} |", "",
        "比较前按样本 ID 对齐。得分通过 CSV 的十进制往返解析保留原数值；同时报告逐值相等和 "
        "`rtol=0, atol=1e-12` 的容差比较，不能把浮点近似一致写成完全相同。", "",
        "## 指标复算", "", "肿瘤为正类 1，正常为负类 0。", "",
        "| 指标 | 原预测 | 复现预测 |", "|---|---:|---:|"]
    for row in metrics.to_dict("records"):
        key = row['metric'];name = METRICS.get(key, key.upper())
        fmt = (lambda x: str(int(x))) if key in ['tn', 'fp', 'fn', 'tp'] else (lambda x: f"{x:.12f}")
        lines.append(f"| {name} | {fmt(row['original'])} | {fmt(row['reproduced'])} |")
    lines += ["", "## 对已有图表的影响", ""]
    if s['status'] == 'passed':
        lines += ["本次核对通过，现有随机森林、SVM 及两者对比图表继续使用，无需重画。"
                  "评价仍是同一批 243 个样本，包含 25 个正常组织和 218 个肿瘤组织。"]
    else:
        lines += ["核对发现差异。现有图表不能直接代表这份复现结果，需要先核实输入来源，"
                  "再使用复现预测在新的输出目录生成评价，保留原图表。"]
    lines += ["", "## 记录范围", "",
        f"本次复现参数为 `{s['reproduced_parameters']}`，选择 {s['feature_count']} 个基因。参数文件中的冻结时间属于运行记录，"
        "与旧文件不同不构成模型参数变化。仅凭这三个文件不能独立确认整个训练过程，"
        "本记录确认的是交付文件的一致性及指标可复算性。", "",
        "逐样本结果、指标对照和输入文件摘要分别保存在 "
        "`results/svm_reproduce_v1_verification/sample_comparison.csv`、`metrics_comparison.csv` 和 `summary.json`。"
        "本记录对应摘要所绑定的项目已有复现文件；其他运行生成的预测应另行核对。", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))


def run_check(original_dir, reproduced_dir, output, report):
    original_dir, reproduced_dir, output, report = [Path(p).resolve() for p in [original_dir, reproduced_dir, output, report]]
    if original_dir == reproduced_dir:
        raise ValueError("Original and reproduced directories must be different")
    for p in [output, report]:
        if p.exists():raise FileExistsError(f"Output already exists: {p}; use a new output path")
    if report == output or report.is_relative_to(output):
        raise ValueError("Keep report and output directory separate")
    original_csv, new_csv = [p/'test_predictions.csv' for p in [original_dir,reproduced_dir]]
    old,new = read_predictions(original_csv),read_predictions(new_csv)
    reference = load_reference(DATA_DIR/'test_data.npz')
    if len(reference)!=243:raise ValueError("Expected the frozen 243-sample test set")
    _,split = validate_manifest(DATA_DIR/'sample_manifest.tsv', reference, DATA_DIR/'train_data.npz')
    summary,detail,a,b = compare_tables(old,new,reference)
    reported = json.loads((reproduced_dir/'test_metrics.json').read_text(encoding='utf-8-sig'))
    before = compute_metrics(a,'SVM');after = validate_reported_metrics(reported,b)
    params = [json.loads((p/'selected_parameters.json').read_text(encoding='utf-8-sig')) for p in [original_dir,reproduced_dir]]
    fields = ['candidate_id','parameters','feature_count','mean_cv','std_cv','test_npz_loaded']
    equal = all(params[0][key]==params[1][key] for key in fields)
    keys = [*METRICS,'tn','fp','fn','tp']
    metrics = pd.DataFrame([{'metric':k,'original':before[k],'reproduced':after[k],
                            'absolute_difference':abs(before[k]-after[k])} for k in keys])
    all_metrics_equal=bool((metrics.absolute_difference<=1e-12).all())
    summary.update({'status':'passed' if summary['prediction_consistency']!='different' and equal and all_metrics_equal else 'different',
        'selected_parameters_equal':equal,'reported_metrics_verified':True,'metrics_equal':all_metrics_equal,
        'prediction_files_byte_identical':digest(original_csv)==digest(new_csv),
        'original_parameters':params[0]['parameters'],'reproduced_parameters':params[1]['parameters'],
        'feature_count':params[1]['feature_count'],
        'original_frozen_at_utc':params[0].get('frozen_at_utc'),'reproduced_frozen_at_utc':params[1].get('frozen_at_utc'),
        'split':split,'training_performed':False,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'environment':{'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn']}},
        'inputs':{path_label(p):digest(p) for p in [original_csv,new_csv,original_dir/'selected_parameters.json',
            reproduced_dir/'selected_parameters.json',reproduced_dir/'test_metrics.json',DATA_DIR/'test_data.npz',
            DATA_DIR/'train_data.npz',DATA_DIR/'sample_manifest.tsv']},'checker_sha256':digest(Path(__file__))})
    output.mkdir(parents=True,exist_ok=False)
    detail.to_csv(output/'sample_comparison.csv',index=False,float_format='%.17g')
    metrics.to_csv(output/'metrics_comparison.csv',index=False,float_format='%.17g')
    write_json(output/'summary.json',summary)
    write_report(report,summary,metrics)
    print(json.dumps({k:summary[k] for k in ['status','n_test_samples','prediction_consistency',
          'prediction_files_byte_identical','max_absolute_score_difference']},ensure_ascii=False,indent=2))
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-dir',type=Path,default=ROOT/'results/svm')
    parser.add_argument('--reproduced-dir',type=Path,default=ROOT/'results/svm_reproduce_v1')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/svm_reproduce_v1_verification')
    parser.add_argument('--report',type=Path,default=ROOT/'docs/svm_reproduce_v1_verification.md')
    args=parser.parse_args(argv)
    try:s=run_check(args.original_dir,args.reproduced_dir,args.output_dir,args.report)
    except (ValueError,OSError,KeyError) as exc:
        print(f'SVM file check failed: {exc}',file=sys.stderr);return 1
    return 0 if s['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
