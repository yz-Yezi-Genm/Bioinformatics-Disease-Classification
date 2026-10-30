# -*- coding: utf-8 -*-
"""
TCGA-BRCA 样本标签整理（角色 B）

输入：成员 A 提供的 UCSC Xena / GDC TCGA-BRCA 原始资源包
    TCGA-BRCA.clinical.tsv.gz   样本临床/注释表
    TCGA-BRCA.star_counts.tsv.gz RNA-seq 表达矩阵（仅读取表头用于样本 ID 核对）
输出：
    data/raw/sample_labels.csv          二分类标签表（肿瘤 1 / 正常 0）
    data/raw/sample_metadata.csv        分析队列样本注释表（含患者信息）
    results/tables/sample_type_counts.csv
    results/tables/patient_level_summary.csv
    results/tables/excluded_samples.csv
    results/tables/clinical_not_in_expression.csv
    results/stats.json                  供文档引用的统计结果

用法：
    python src/build_labels.py --raw "C:\\path\\to\\TCGA-BRCA-data"
"""
import argparse
import collections
import csv
import gzip
import json
import os

SAMPLE_TYPE_MAP = {
    "01": ("Primary Tumor", 1),
    "11": ("Solid Tissue Normal", 0),
}
EXCLUDED_TYPE_CODE = {"06": "Metastatic"}


def read_tsv_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = [r for r in reader if r]
    return header, rows


def read_expr_header(path):
    """只读表达矩阵第一行，拿到样本条码列表。"""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
    return header[0], header[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=r"C:\Users\asus\Downloads\TCGA-BRCA-data")
    ap.add_argument("--out", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()

    out = args.out
    for d in ("data/raw", "results/tables"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    clin_path = os.path.join(args.raw, "TCGA-BRCA.clinical.tsv.gz")
    expr_path = os.path.join(args.raw, "TCGA-BRCA.star_counts.tsv.gz")

    header, rows = read_tsv_gz(clin_path)
    idx = {h: i for i, h in enumerate(header)}
    expr_gene_col, expr_samples = read_expr_header(expr_path)

    def get(row, col):
        v = row[idx[col]]
        return "" if v in ("", "NA", "N/A", "null") else v

    # ---------- 1. 读取临床表，解析条码 ----------
    records = []
    for row in rows:
        sid = row[idx["sample"]].strip()
        code = sid[13:15]
        if code in SAMPLE_TYPE_MAP:
            label = SAMPLE_TYPE_MAP[code][1]
            in_binary = True
        else:
            label = None
            in_binary = False
        records.append({
            "sample_id": sid,
            "patient_id": sid[:12],
            "sample_type_code": code,
            "sample_type": get(row, "sample_type.samples"),
            "tumor_descriptor": get(row, "tumor_descriptor.samples"),
            "specimen_type": get(row, "specimen_type.samples"),
            "tissue_type": get(row, "tissue_type.samples"),
            "is_ffpe": get(row, "is_ffpe.samples"),
            "gender": get(row, "gender.demographic"),
            "age_at_index": get(row, "age_at_index.demographic"),
            "vital_status": get(row, "vital_status.demographic"),
            "ajcc_pathologic_stage": get(row, "ajcc_pathologic_stage.diagnoses"),
            "primary_diagnosis": get(row, "primary_diagnosis.diagnoses"),
            "days_to_death": get(row, "days_to_death.demographic"),
            "days_to_last_follow_up": get(row, "days_to_last_follow_up.diagnoses"),
            "sample_type_clinical": get(row, "sample_type.samples"),
            "sample_type_id_clinical": get(row, "sample_type_id.samples"),
            "label": label,
            "in_binary_task": in_binary,
            "_clinical_type": get(row, "sample_type.samples"),
        })

    clin_by_id = {r["sample_id"]: r for r in records}
    expr_set = set(expr_samples)

    # ---------- 2. 核对：条码类型码 vs clinical 列 ----------
    code_mismatch = [
        r["sample_id"] for r in records
        if r["sample_type_id_clinical"] != r["sample_type_code"]
    ]
    type_col_mismatch = [
        r["sample_id"] for r in records
        if r["sample_type_code"] in SAMPLE_TYPE_MAP
        and r["sample_type_clinical"] != SAMPLE_TYPE_MAP[r["sample_type_code"]][0]
    ]

    # patient_id 与 case submitter id 是否一致
    patient_mismatch = []
    for row in rows:
        sid = row[idx["sample"]].strip()
        if sid[:12] != row[idx["submitter_id"]].strip():
            patient_mismatch.append(sid)

    # 重复样本
    id_counts = collections.Counter(r["sample_id"] for r in records)
    dup_ids = [k for k, v in id_counts.items() if v > 1]

    # ---------- 3. 与 A 的表达矩阵取交集 ----------
    cohort = [r for r in records if r["sample_id"] in expr_set]
    not_in_expr = [r for r in records if r["sample_id"] not in expr_set]
    expr_not_in_clin = sorted(expr_set - set(clin_by_id))

    # ---------- 4. 统计 ----------
    type_counts_all = collections.Counter(r["sample_type"] for r in records)
    type_counts_cohort = collections.Counter(r["sample_type"] for r in cohort)
    binary = [r for r in cohort if r["in_binary_task"]]
    tumor = [r for r in binary if r["label"] == 1]
    normal = [r for r in binary if r["label"] == 0]
    excluded = [r for r in cohort if not r["in_binary_task"]]

    # 缺失标签检查
    missing_label = [r["sample_id"] for r in cohort if r["label"] is None]
    missing_type = [r["sample_id"] for r in cohort if not r["sample_type"]]

    # 患者层面
    pat = collections.defaultdict(list)
    for r in cohort:
        pat[r["patient_id"]].append(r)
    binary_pat = collections.defaultdict(list)
    for r in binary:
        binary_pat[r["patient_id"]].append(r)
    pat_both = {p: v for p, v in binary_pat.items()
                if {x["label"] for x in v} == {0, 1}}
    pat_multi_tumor = {p: [x["sample_id"] for x in v if x["label"] == 1]
                       for p, v in binary_pat.items()
                       if sum(1 for x in v if x["label"] == 1) > 1}
    pat_multi_sample = {p: [x["sample_id"] for x in v] for p, v in binary_pat.items()
                        if len(v) > 1}
    normals_without_tumor = sorted(
        {p for p, v in binary_pat.items()
         if all(x["label"] == 0 for x in v)}
    )

    multi_sample_clinical = sum(1 for p, v in pat.items() if len(v) > 1)

    stats = {
        "raw_dir": args.raw,
        "clinical_rows": len(records),
        "clinical_unique_samples": len(clin_by_id),
        "expression_samples": len(expr_samples),
        "expression_gene_col": expr_gene_col,
        "matched_samples": len(cohort),
        "clinical_only": len(not_in_expr),
        "expression_only": len(expr_not_in_clin),
        "type_counts_all": dict(type_counts_all),
        "type_counts_cohort": dict(type_counts_cohort),
        "binary_total": len(binary),
        "tumor_n": len(tumor),
        "normal_n": len(normal),
        "tumor_normal_ratio": round(len(tumor) / len(normal), 2),
        "excluded_n": len(excluded),
        "excluded_by_type": dict(collections.Counter(r["sample_type"] for r in excluded)),
        "missing_label_n": len(missing_label),
        "missing_type_n": len(missing_type),
        "duplicate_ids": dup_ids,
        "code_mismatch": len(code_mismatch),
        "type_col_mismatch": len(type_col_mismatch),
        "patient_mismatch": len(patient_mismatch),
        "patients_cohort": len(pat),
        "patients_binary": len(binary_pat),
        "patients_both_classes": len(pat_both),
        "patients_multi_tumor": len(pat_multi_tumor),
        "patients_multi_sample_cohort": multi_sample_clinical,
        "patients_multi_sample_binary": len(pat_multi_sample),
        "normals_without_tumor_patients": normals_without_tumor,
        "ffpe_n": sum(1 for r in cohort if r["is_ffpe"] == "True"),
    }

    # ---------- 5. 写文件 ----------
    label_cols = ["sample_id", "patient_id", "sample_type", "sample_type_code", "label"]
    with open(os.path.join(out, "data/raw/sample_labels.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(label_cols)
        for r in sorted(binary, key=lambda x: x["sample_id"]):
            w.writerow([r["sample_id"], r["patient_id"], r["sample_type"],
                        r["sample_type_code"], r["label"]])

    meta_cols = ["sample_id", "patient_id", "sample_type", "sample_type_code",
                 "tumor_descriptor", "specimen_type", "tissue_type", "is_ffpe",
                 "gender", "age_at_index", "vital_status", "ajcc_pathologic_stage",
                 "primary_diagnosis", "days_to_death", "days_to_last_follow_up",
                 "label", "in_binary_task"]
    with open(os.path.join(out, "data/raw/sample_metadata.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(meta_cols)
        for r in sorted(cohort, key=lambda x: x["sample_id"]):
            w.writerow([r[c] if r[c] is not None else "" for c in meta_cols])

    with open(os.path.join(out, "results/tables/sample_type_counts.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_type", "sample_type_code", "clinical_rows",
                    "in_expression_matrix", "label", "in_binary_task"])
        for st in ["Primary Tumor", "Solid Tissue Normal", "Metastatic"]:
            code = {"Primary Tumor": "01", "Solid Tissue Normal": "11",
                    "Metastatic": "06"}[st]
            lab = {1: "1", 0: "0"}.get(SAMPLE_TYPE_MAP.get(code, (None, None))[1], "")
            w.writerow([st, code, type_counts_all.get(st, 0),
                        type_counts_cohort.get(st, 0), lab,
                        "Yes" if code in SAMPLE_TYPE_MAP else "No"])

    with open(os.path.join(out, "results/tables/patient_level_summary.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["指标", "数值"])
        rows_out = [
            ("分析队列患者数", len(pat)),
            ("二分类患者数", len(binary_pat)),
            ("同时含肿瘤与正常样本的患者数", len(pat_both)),
            ("含 ≥2 个肿瘤样本的患者数", len(pat_multi_tumor)),
            ("含 ≥2 个样本的二分类患者数", len(pat_multi_sample)),
            ("仅有正常样本、无肿瘤样本的患者数", len(normals_without_tumor)),
            ("二分类样本数", len(binary)),
            ("其中肿瘤样本数", len(tumor)),
            ("其中正常样本数", len(normal)),
            ("肿瘤:正常", f"{len(tumor)}:{len(normal)}"),
        ]
        for k, v in rows_out:
            w.writerow([k, v])

    with open(os.path.join(out, "results/tables/excluded_samples.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "patient_id", "sample_type", "sample_type_code",
                    "exclusion_reason"])
        for r in sorted(excluded, key=lambda x: x["sample_id"]):
            w.writerow([r["sample_id"], r["patient_id"], r["sample_type"],
                        r["sample_type_code"],
                        "非肿瘤/正常二分类类型（转移灶），按任务要求单列"])

    with open(os.path.join(out, "results/tables/clinical_not_in_expression.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "patient_id", "sample_type", "sample_type_code",
                    "note"])
        for r in sorted(not_in_expr, key=lambda x: x["sample_id"]):
            w.writerow([r["sample_id"], r["patient_id"], r["sample_type"],
                        r["sample_type_code"],
                        "临床表有记录，但 A 的表达矩阵中无此样本，无法纳入建模"])

    with open(os.path.join(out, "results/stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\npatients_both_classes ids:", sorted(pat_both)[:15])
    print("multi tumor:", dict(list(pat_multi_tumor.items())[:10]))
    print("normals without tumor:", normals_without_tumor)


if __name__ == "__main__":
    main()
