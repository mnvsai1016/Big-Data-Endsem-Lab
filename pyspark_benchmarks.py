import os
import sys
import time
import json
import jdk4py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Clean PATH
py312_dir = r"C:\Users\Mnvsai\AppData\Local\Programs\Python\Python312"
py312_scripts = r"C:\Users\Mnvsai\AppData\Local\Programs\Python\Python312\Scripts"
clean_paths = [py312_dir, py312_scripts]

for p in os.environ.get("PATH", "").split(os.pathsep):
    if "pythoncore-3.14" in p or r"AppData\Local\Python\bin" in p:
        continue
    if p not in clean_paths:
        clean_paths.append(p)

os.environ["PATH"] = os.pathsep.join(clean_paths)
os.environ['JAVA_HOME'] = str(jdk4py.JAVA_HOME)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.fpm import FPGrowth as SparkFPGrowth

from sklearn.cluster import KMeans as SklearnKMeans
from mlxtend.frequent_patterns import apriori as mlx_apriori

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Initialize Spark Session on all 20 cores
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("PySpark_BigData_Benchmarks") \
    .config("spark.pyspark.python", sys.executable) \
    .config("spark.pyspark.driver.python", sys.executable) \
    .config("spark.driver.memory", "6g") \
    .config("spark.ui.enabled", "false") \
    .config("spark.sql.shuffle.partitions", "20") \
    .getOrCreate()

cores = spark.sparkContext.defaultParallelism
print(f"SparkSession Active on {cores} CPU cores (Java 17 via jdk4py)")

# Load base dataset
base_df = pd.read_csv("datasets/fifa21_male2.csv", low_memory=False)
base_n = len(base_df)
print(f"Base dataset size: {base_n} rows")

# Prepare features for single-node
base_stats = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']
for c in base_stats:
    base_df[c] = pd.to_numeric(base_df[c], errors='coerce').fillna(base_df[c].median())

# Binary ARM items for single-node
arm_base = pd.DataFrame()
for col in base_stats:
    arm_base[f"High_{col}"] = base_df[col] >= base_df[col].median()

# Scale factors: 1x (17K), 2x (34K), 5x (85K), 10x (171K)
scale_factors = [1, 2, 5, 10]
row_counts = [base_n * s for s in scale_factors]

# Benchmark result structures
fpm_results = []
kmeans_results = []

print("\n" + "=" * 60)
print("BENCHMARK 1: PYSPARK FP-GROWTH VS SINGLE-NODE APRIORI")
print("=" * 60)

for s in scale_factors:
    n_rows = base_n * s
    print(f"\nEvaluating Scale Factor {s}x ({n_rows:,} rows)...")
    
    # 1. Single-node Apriori
    arm_scaled = pd.concat([arm_base] * s, ignore_index=True)
    t0 = time.time()
    fi_single = mlx_apriori(arm_scaled, min_support=0.15, use_colnames=True)
    t_single_apriori = time.time() - t0
    print(f"  Single-node Apriori:    {t_single_apriori:6.3f}s (FIs found: {len(fi_single)})")
    
    # 2. PySpark MLlib FP-Growth
    # Read/union Spark DataFrame to scale
    base_spark = spark.read.csv("datasets/fifa21_male2.csv", header=True, inferSchema=True)
    if s == 1:
        scaled_spark = base_spark
    else:
        dfs = [base_spark] * s
        scaled_spark = dfs[0]
        for extra in dfs[1:]:
            scaled_spark = scaled_spark.unionAll(extra)
            
    # Items expression
    items_expr = F.array_compact(F.array([
        F.when(F.col(c) >= float(base_df[c].median()), F.lit(f"High_{c}")).otherwise(F.lit(None))
        for c in base_stats
    ]))
    
    basket_df = scaled_spark.withColumn("items", items_expr).select("items").cache()
    _ = basket_df.count() # Force cache materialize
    
    t0 = time.time()
    fp = SparkFPGrowth(itemsCol="items", minSupport=0.15, minConfidence=0.7)
    fp_model = fp.fit(basket_df)
    n_spark_fi = fp_model.freqItemsets.count()
    t_spark_fpgrowth = time.time() - t0
    print(f"  PySpark Parallel FPM:   {t_spark_fpgrowth:6.3f}s (FIs found: {n_spark_fi})")
    
    speedup_fpm = t_single_apriori / t_spark_fpgrowth if t_spark_fpgrowth > 0 else 1.0
    print(f"  -> FP-Growth Speedup:   {speedup_fpm:.2f}x")
    
    fpm_results.append({
        "scale": s,
        "rows": n_rows,
        "single_node_apriori_sec": round(t_single_apriori, 3),
        "pyspark_fpgrowth_sec": round(t_spark_fpgrowth, 3),
        "speedup": round(speedup_fpm, 2)
    })
    basket_df.unpersist()

print("\n" + "=" * 60)
print("BENCHMARK 2: PYSPARK DISTRIBUTED K-MEANS VS SCIKIT-LEARN")
print("=" * 60)

for s in scale_factors:
    n_rows = base_n * s
    print(f"\nEvaluating Scale Factor {s}x ({n_rows:,} rows)...")
    
    # 1. Scikit-Learn KMeans
    X_scaled_np = np.tile(base_df[base_stats].values, (s, 1))
    t0 = time.time()
    sk_km = SklearnKMeans(n_clusters=4, random_state=42, n_init=5, max_iter=20)
    sk_km.fit(X_scaled_np)
    t_sklearn_kmeans = time.time() - t0
    print(f"  Scikit-Learn KMeans:    {t_sklearn_kmeans:6.3f}s")
    
    # 2. PySpark MLlib KMeans
    base_spark = spark.read.csv("datasets/fifa21_male2.csv", header=True, inferSchema=True)
    if s == 1:
        scaled_spark = base_spark
    else:
        dfs = [base_spark] * s
        scaled_spark = dfs[0]
        for extra in dfs[1:]:
            scaled_spark = scaled_spark.unionAll(extra)
            
    assembler = VectorAssembler(inputCols=base_stats, outputCol="features")
    vec_df = assembler.transform(scaled_spark.select(base_stats).dropna()).cache()
    _ = vec_df.count()
    
    t0 = time.time()
    spark_km = SparkKMeans(k=4, seed=42, maxIter=20)
    spark_km_model = spark_km.fit(vec_df)
    t_spark_kmeans = time.time() - t0
    print(f"  PySpark MLlib KMeans:   {t_spark_kmeans:6.3f}s")
    
    speedup_kmeans = t_sklearn_kmeans / t_spark_kmeans if t_spark_kmeans > 0 else 1.0
    print(f"  -> KMeans Speedup:      {speedup_kmeans:.2f}x")
    
    kmeans_results.append({
        "scale": s,
        "rows": n_rows,
        "sklearn_kmeans_sec": round(t_sklearn_kmeans, 3),
        "pyspark_kmeans_sec": round(t_spark_kmeans, 3),
        "speedup": round(speedup_kmeans, 2)
    })
    vec_df.unpersist()

# Stop Spark
spark.stop()

# Plot Figure 15: PySpark Scaling Curves
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

rows_k = [r / 1000 for r in row_counts]

# Plot 1: ARM execution time
ax1.plot(rows_k, [r['single_node_apriori_sec'] for r in fpm_results], marker='o', color='#d62728', linewidth=2, label='Single-Node Apriori')
ax1.plot(rows_k, [r['pyspark_fpgrowth_sec'] for r in fpm_results], marker='s', color='#1f77b4', linewidth=2, label='PySpark FP-Growth')
ax1.set_xlabel("Dataset Volume (Thousands of Rows)", fontweight='bold')
ax1.set_ylabel("Execution Time (Seconds)", fontweight='bold')
ax1.set_title("ARM Execution Time Scaling", fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot 2: KMeans execution time
ax2.plot(rows_k, [r['sklearn_kmeans_sec'] for r in kmeans_results], marker='o', color='#ff7f0e', linewidth=2, label='Scikit-Learn KMeans')
ax2.plot(rows_k, [r['pyspark_kmeans_sec'] for r in kmeans_results], marker='s', color='#2ca02c', linewidth=2, label='PySpark MLlib KMeans')
ax2.set_xlabel("Dataset Volume (Thousands of Rows)", fontweight='bold')
ax2.set_ylabel("Execution Time (Seconds)", fontweight='bold')
ax2.set_title("K-Means Execution Time Scaling", fontsize=11, fontweight='bold')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

# Plot 3: Speedup curves & crossover
ax3.plot(rows_k, [r['speedup'] for r in fpm_results], marker='^', color='#1f77b4', linewidth=2.5, label='FP-Growth Speedup')
ax3.plot(rows_k, [r['speedup'] for r in kmeans_results], marker='D', color='#2ca02c', linewidth=2.5, label='K-Means Speedup')
ax3.axhline(1.0, color='gray', linestyle=':', label='Parity (1.0x)')
ax3.set_xlabel("Dataset Volume (Thousands of Rows)", fontweight='bold')
ax3.set_ylabel("Distributed Speedup Factor (x)", fontweight='bold')
ax3.set_title("Distributed Scalability & Crossover Point", fontsize=11, fontweight='bold')
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.5)

plt.suptitle("PySpark 3.5.5 Distributed Big Data Performance Benchmarks (20 CPU Cores)", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("report_images/fig15_pyspark_scaling_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved report_images/fig15_pyspark_scaling_curves.png")

# Save JSON results
benchmark_summary = {
    "cores": cores,
    "spark_version": "3.5.5",
    "java_home": str(jdk4py.JAVA_HOME),
    "fpm_benchmarks": fpm_results,
    "kmeans_benchmarks": kmeans_results
}

with open("pyspark_benchmark_results.json", "w") as f:
    json.dump(benchmark_summary, f, indent=2)

print("Saved pyspark_benchmark_results.json successfully!")
