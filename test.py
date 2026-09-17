from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pyarrow.types as pa_types


# ============================================================
# 1. 修改成你自己的 parquet 文件路径
# ============================================================

PARQUET_PATH = Path("data/raw/compound_profiles.parquet")

# 统计结果会保存到这里
REPORT_PATH = Path("parquet_report.txt")


# ============================================================
# 2. 用来同时打印并保存结果
# ============================================================

report = []


def log(content=""):
    content = str(content)
    print(content)
    report.append(content)


# ============================================================
# 3. 读取 parquet 的基本结构
# ============================================================

parquet_file = pq.ParquetFile(PARQUET_PATH)

schema = parquet_file.schema_arrow

all_columns = schema.names

row_number = parquet_file.metadata.num_rows
column_number = len(all_columns)


log("=" * 60)
log("BASIC INFORMATION")
log("=" * 60)

log(f"file: {PARQUET_PATH}")
log(f"rows: {row_number}")
log(f"columns: {column_number}")
log(f"row groups: {parquet_file.num_row_groups}")


# ============================================================
# 4. 区分 Metadata 和 phenotype feature
# ============================================================

metadata_columns = []

feature_columns = []

for column in all_columns:

    if column.startswith("Metadata_"):
        metadata_columns.append(column)

    else:
        feature_columns.append(column)


log()
log("=" * 60)
log("COLUMN INFORMATION")
log("=" * 60)

log(f"metadata column number: {len(metadata_columns)}")
log(f"feature column number: {len(feature_columns)}")

log()
log("metadata columns:")

for column in metadata_columns:
    log(column)


log()
log("first 20 feature columns:")

for column in feature_columns[:20]:
    log(column)


log()
log("last 20 feature columns:")

for column in feature_columns[-20:]:
    log(column)


# ============================================================
# 5. 检查 phenotype feature 的数据类型
# ============================================================

numeric_feature_number = 0
non_numeric_features = []

for column in feature_columns:

    column_type = schema.field(column).type

    if (
        pa_types.is_integer(column_type)
        or pa_types.is_floating(column_type)
    ):
        numeric_feature_number += 1

    else:
        non_numeric_features.append(
            (column, str(column_type))
        )


log()
log("=" * 60)
log("FEATURE DATA TYPES")
log("=" * 60)

log(f"numeric feature number: {numeric_feature_number}")
log(f"non-numeric feature number: {len(non_numeric_features)}")


if len(non_numeric_features) > 0:

    log()
    log("non-numeric features:")

    for column, dtype in non_numeric_features:
        log(f"{column}: {dtype}")


# ============================================================
# 6. 只读取 Metadata
# ============================================================

log()
log("=" * 60)
log("READING METADATA")
log("=" * 60)

metadata_df = pd.read_parquet(
    PARQUET_PATH,
    columns=metadata_columns
)

log(f"metadata shape: {metadata_df.shape}")


# ============================================================
# 7. 查看前 5 行 Metadata
# ============================================================

log()
log("=" * 60)
log("FIRST 5 METADATA ROWS")
log("=" * 60)

log(metadata_df.head().to_string())


# ============================================================
# 8. 每一个 Metadata 列有多少种不同的值
# ============================================================

log()
log("=" * 60)
log("METADATA UNIQUE COUNTS")
log("=" * 60)

for column in metadata_columns:

    unique_number = metadata_df[column].nunique(
        dropna=True
    )

    missing_number = metadata_df[column].isna().sum()

    log(
        f"{column}: "
        f"unique={unique_number}, "
        f"missing={missing_number}"
    )


# ============================================================
# 9. 自动寻找可能的 compound ID 列
# ============================================================

id_candidates = []

keywords = [
    "jcp",
    "compound",
    "perturb",
    "pert_"
]

for column in metadata_columns:

    lower_name = column.lower()

    for keyword in keywords:

        if keyword in lower_name:

            id_candidates.append(column)
            break


log()
log("=" * 60)
log("POSSIBLE COMPOUND ID COLUMNS")
log("=" * 60)

if len(id_candidates) == 0:

    log("No obvious compound ID column found.")

else:

    for column in id_candidates:
        log(column)


# ============================================================
# 10. 优先分析 Metadata_JCP2022
# ============================================================

if "Metadata_JCP2022" in metadata_df.columns:

    compound_id_column = "Metadata_JCP2022"

else:

    compound_id_column = None

    if len(id_candidates) > 0:
        compound_id_column = id_candidates[0]


# ============================================================
# 11. 分析一个 compound 对应多少 phenotype
# ============================================================

if compound_id_column is not None:

    log()
    log("=" * 60)
    log("COMPOUND / PHENOTYPE RELATIONSHIP")
    log("=" * 60)

    log(f"compound ID column used: {compound_id_column}")

    compound_series = metadata_df[
        compound_id_column
    ].dropna()

    counts = compound_series.value_counts()

    log(
        f"non-null rows: "
        f"{len(compound_series)}"
    )

    log(
        f"unique compounds: "
        f"{compound_series.nunique()}"
    )

    log(
        f"compounds appearing more than once: "
        f"{(counts > 1).sum()}"
    )

    log(
        f"compounds appearing only once: "
        f"{(counts == 1).sum()}"
    )

    log()
    log("phenotype profiles per compound:")

    log(
        counts.describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        ).to_string()
    )

    log()
    log("20 compounds with the most profiles:")

    log(
        counts.head(20).to_string()
    )


# ============================================================
# 12. 查看实验层级相关 Metadata
# ============================================================

experiment_keywords = [
    "plate",
    "well",
    "source",
    "batch",
    "site"
]

experiment_columns = []

for column in metadata_columns:

    lower_name = column.lower()

    for keyword in experiment_keywords:

        if keyword in lower_name:

            experiment_columns.append(column)
            break


log()
log("=" * 60)
log("EXPERIMENT-RELATED METADATA")
log("=" * 60)

for column in experiment_columns:

    log(
        f"{column}: "
        f"{metadata_df[column].nunique(dropna=True)} unique values"
    )


# ============================================================
# 13. 保存结果
# ============================================================

REPORT_PATH.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print()
print("=" * 60)
print(f"Report saved to: {REPORT_PATH.resolve()}")
print("=" * 60)
