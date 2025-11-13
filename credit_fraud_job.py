import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame, DynamicFrameCollection
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import pandas as pd
from sklearn.model_selection import KFold

# ============================================================
# 1. COMBINED FEATURE ENGINEERING + MERCHANT OHE + JOB TARGET ENCODING
# ============================================================
def FeatureEngineering(glueContext, dfc):
    df = dfc.select(list(dfc.keys())[0]).toDF()

    # -------------------------------
    # Cast event_time
    # -------------------------------
    df = df.withColumn("event_time", F.col("event_time").cast("timestamp"))

    # -------------------------------
    # Rolling windows based on city
    # -------------------------------
    w7  = Window.partitionBy("city").orderBy(F.col("event_time").cast("long")).rangeBetween(-7*86400, -1)
    w30 = Window.partitionBy("city").orderBy(F.col("event_time").cast("long")).rangeBetween(-30*86400, -1)
    w90 = Window.partitionBy("city").orderBy(F.col("event_time").cast("long")).rangeBetween(-90*86400, -1)

    df = df.withColumn("city_avg_amt_7d",  F.avg("amt").over(w7)) \
           .withColumn("city_avg_amt_30d", F.avg("amt").over(w30)) \
           .withColumn("city_avg_amt_90d", F.avg("amt").over(w90)) \
           .withColumn("city_fraud_rate_30d", F.avg(F.col("is_fraud").cast("double")).over(w30)) \
           .withColumn("city_fraud_rate_90d", F.avg(F.col("is_fraud").cast("double")).over(w90))

    # -------------------------------
    # Time since last transaction per user
    # -------------------------------
    w_user = Window.partitionBy("user_id").orderBy("event_time")
    df = df.withColumn("time_since_last_txn",
                       F.col("event_time").cast("long") -
                       F.lag(F.col("event_time").cast("long")).over(w_user))

    df = df.fillna({
        "city_avg_amt_7d": 0.0,
        "city_avg_amt_30d": 0.0,
        "city_avg_amt_90d": 0.0,
        "city_fraud_rate_30d": 0.0,
        "city_fraud_rate_90d": 0.0,
        "time_since_last_txn": 0.0
    })

    # ============================================================
    # 2. MERCHANT ONE-HOT
    # ============================================================
    merchants = [r["merchant"] for r in df.select("merchant").distinct().collect()]

    for m in merchants:
        clean_name = m.replace(" ", "_").replace("-", "_").replace("/", "_")
        df = df.withColumn(f"merchant_{clean_name}",
                           F.when(F.col("merchant") == m, 1).otherwise(0))

    # ============================================================
    # 3. JOB GROUPING
    # ============================================================
    def group_job(job):
        j = str(job).lower().strip()
        if any(k in j for k in ["engineer","developer","programmer","it","systems","network","data","scientist"]):
            return "tech_engineering"
        elif any(k in j for k in ["nurse","doctor","pharmacist","therapist","psychologist","biomedical","clinical","surgeon","dentist"]):
            return "healthcare"
        elif any(k in j for k in ["teacher","professor","lecturer","tutor","education","academic","librarian"]):
            return "education"
        elif any(k in j for k in ["accountant","bank","finance","financial","economist",
                                  "cfo","ceo","manager","risk","insurance","tax","auditor","trader"]):
            return "finance_business"
        elif any(k in j for k in ["designer","artist","musician","writer","journalist","editor",
                                  "actor","curator","photographer","director"]):
            return "arts_media"
        elif any(k in j for k in ["lawyer","solicitor","barrister","legal","police","civil",
                                  "government","diplomatic","military"]):
            return "law_government"
        elif any(k in j for k in ["sales","marketing","customer","retail","hospitality",
                                  "hotel","tour","restaurant","public relations"]):
            return "sales_service"
        elif any(k in j for k in ["technician","horticultur","farmer","craft","construction","builder","technologist"]):
            return "skilled_trades"
        else:
            return "other"

    group_udf = udf(group_job, StringType())
    df = df.withColumn("job_grouped", group_udf(F.col("job")))

    # ============================================================
    # 4. JOB TARGET ENCODING (KFold + smoothing)
    # ============================================================
    pdf = df.select("job_grouped", "is_fraud").toPandas()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = pdf["is_fraud"].mean()

    encoded = pd.Series(index=pdf.index, dtype=float)

    for train_idx, val_idx in kf.split(pdf):
        train = pdf.iloc[train_idx]
        means = train.groupby("job_grouped")["is_fraud"].mean()
        counts = train["job_grouped"].value_counts()
        smoothing = 50
        smooth = (means * counts + global_mean * smoothing) / (counts + smoothing)

        encoded.iloc[val_idx] = pdf.iloc[val_idx]["job_grouped"].map(smooth).fillna(global_mean)

    pdf["job_target_enc"] = encoded

    enc_df = spark.createDataFrame(pdf[["job_target_enc"]].reset_index())

    df_idx = df.rdd.zipWithIndex().toDF(["row", "idx"])
    enc_idx = enc_df.rdd.zipWithIndex().toDF(["row2", "idx"])

    df = df_idx.join(enc_idx, "idx").select(
        F.col("row.*"),
        F.col("row2.job_target_enc").alias("job")
    )

    # ============================================================
    # 5. DROP UNUSED FIELDS
    # ============================================================
    df = df.drop("unix_time", "first", "last", "merchant", "job_grouped")

    return DynamicFrameCollection(
        {"FeatureEngineering": DynamicFrame.fromDF(df, glueContext, "FeatureEngineering")},
        glueContext
    )


# ============================================================
# 6. LOG TRANSFORM
# ============================================================
def LogTransform(glueContext, dfc):
    df = dfc.select(list(dfc.keys())[0]).toDF()

    cols = [
        "amt", "city_avg_amt_7d", "city_avg_amt_30d", "city_avg_amt_90d",
        "city_fraud_rate_30d", "city_fraud_rate_90d", "time_since_last_txn"
    ]

    for c in cols:
        df = df.withColumn(f"log_{c}", F.log(F.col(c) + 1))

    return DynamicFrameCollection(
        {"LogTransformed": DynamicFrame.fromDF(df, glueContext, "LogTransformed")},
        glueContext
    )


# ============================================================
# MAIN PIPELINE
# ============================================================
args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Load input
input_df = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    format="csv",
    format_options={"withHeader": True, "separator": ","},
    connection_options={"paths": ["s3://credittransaction-fraud"], "recurse": True},
)

# Pipeline (Only 2 steps now)
step1 = FeatureEngineering(glueContext, DynamicFrameCollection({"input": input_df}, glueContext))
step2 = LogTransform(glueContext, step1)

final_df = step2.select("LogTransformed")

# Write output
glueContext.write_dynamic_frame.from_options(
    frame=final_df,
    connection_type="s3",
    format="csv",
    connection_options={"path": "s3://credittransaction-processed", "partitionKeys": []},
)

job.commit()
