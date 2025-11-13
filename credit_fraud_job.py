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

# ============================================================
# 1. FEATURE ENGINEERING + MERCHANT OHE + JOB TARGET ENCODING (Optimized)
# ============================================================
def FeatureEngineeringOptimized(glueContext, dfc):
    df = dfc.select(list(dfc.keys())[0]).toDF()

    # -------------------------------
    # Cast trans_time
    # -------------------------------
    df = df.withColumn("trans_time", F.col("trans_time").cast("timestamp"))

    # -------------------------------
    # Rolling windows based on city
    # -------------------------------
    w7  = Window.partitionBy("city").orderBy(F.col("trans_time").cast("long")).rangeBetween(-7*86400, -1)
    w30 = Window.partitionBy("city").orderBy(F.col("trans_time").cast("long")).rangeBetween(-30*86400, -1)
    w90 = Window.partitionBy("city").orderBy(F.col("trans_time").cast("long")).rangeBetween(-90*86400, -1)

    df = df.withColumn("city_avg_amt_7d",  F.avg("amt").over(w7)) \
           .withColumn("city_avg_amt_30d", F.avg("amt").over(w30)) \
           .withColumn("city_avg_amt_90d", F.avg("amt").over(w90)) \
           .withColumn("city_fraud_rate_30d", F.avg(F.col("is_fraud").cast("double")).over(w30)) \
           .withColumn("city_fraud_rate_90d", F.avg(F.col("is_fraud").cast("double")).over(w90))

    # -------------------------------
    # Time since last transaction per user (use ssn)
    # -------------------------------
    w_user = Window.partitionBy("ssn").orderBy("trans_time")
    df = df.withColumn("time_since_last_txn",
                       F.col("trans_time").cast("long") -
                       F.lag(F.col("trans_time").cast("long")).over(w_user))

    df = df.fillna({
        "city_avg_amt_7d": 0.0,
        "city_avg_amt_30d": 0.0,
        "city_avg_amt_90d": 0.0,
        "city_fraud_rate_30d": 0.0,
        "city_fraud_rate_90d": 0.0,
        "time_since_last_txn": 0.0
    })

    # ============================================================
    # Merchant One-Hot Encoding (pivot)
    # ============================================================
    df = df.withColumn("merchant_clean", F.regexp_replace(F.regexp_replace(F.regexp_replace(F.col("merchant"), " ", "_"), "-", "_"), "/", "_"))
    merchant_ohe = df.groupBy("ssn").pivot("merchant_clean").agg(F.lit(1)).na.fill(0)
    df = df.join(merchant_ohe, on="ssn", how="left")

    # ============================================================
    # Job Grouping
    # ============================================================
    def group_job(job):
        j = str(job).lower().strip()
        if any(k in j for k in ["engineer","developer","programmer","it","systems","network","data","scientist"]):
            return "tech_engineering"
        elif any(k in j for k in ["nurse","doctor","pharmacist","therapist","psychologist","biomedical","clinical","surgeon","dentist"]):
            return "healthcare"
        elif any(k in j for k in ["teacher","professor","lecturer","tutor","education","academic","librarian"]):
            return "education"
        elif any(k in j for k in ["accountant","bank","finance","financial","economist","cfo","ceo","manager","risk","insurance","tax","auditor","trader"]):
            return "finance_business"
        elif any(k in j for k in ["designer","artist","musician","writer","journalist","editor","actor","curator","photographer","director"]):
            return "arts_media"
        elif any(k in j for k in ["lawyer","solicitor","barrister","legal","police","civil","government","diplomatic","military"]):
            return "law_government"
        elif any(k in j for k in ["sales","marketing","customer","retail","hospitality","hotel","tour","restaurant","public relations"]):
            return "sales_service"
        elif any(k in j for k in ["technician","horticultur","farmer","craft","construction","builder","technologist"]):
            return "skilled_trades"
        else:
            return "other"

    group_udf = udf(group_job, StringType())
    df = df.withColumn("job_grouped", group_udf(F.col("job")))

    # ============================================================
    # Job Target Encoding (Spark only)
    # ============================================================
    job_stats = df.groupBy("job_grouped").agg(F.avg(F.col("is_fraud").cast("double")).alias("job_target_enc"))
    df = df.join(job_stats, on="job_grouped", how="left")

    # ============================================================
    # Drop unused fields
    # ============================================================
    drop_cols = [
    "ssn", "cc_num", "first", "last", "gender",
    "street", "city", "state", "zip", "lat", "long",
    "acct_num", "profile", "trans_num", "trans_date",
    "trans_time", "trans_time_day", "age", "merchant",
    "customer_num_trans_7_day"]

    df = df.drop(*[c for c in drop_cols if c in df.columns])

    return DynamicFrameCollection(
        {"FeatureEngineering": DynamicFrame.fromDF(df, glueContext, "FeatureEngineering")},
        glueContext
    )


# ============================================================
# 2. LOG TRANSFORM
# ============================================================
def LogTransform(glueContext, dfc):
    df = dfc.select(list(dfc.keys())[0]).toDF()

    cols = [
        "amt", "city_avg_amt_7d", "city_avg_amt_30d", "city_avg_amt_90d",
        "city_fraud_rate_30d", "city_fraud_rate_90d", "time_since_last_txn"
    ]

    for c in cols:
        if c in df.columns:
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

# Load input from your bucket
input_df = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    format="csv",
    format_options={"withHeader": True, "separator": ","},
    connection_options={"paths": ["s3://credit-transaction-fruad-new"], "recurse": False},
)

step1 = FeatureEngineeringOptimized(glueContext, DynamicFrameCollection({"input": input_df}, glueContext))

# Run Log Transform
step2 = LogTransform(glueContext, step1)
final_df = step2.select("LogTransformed")

# Write output back to S3
glueContext.write_dynamic_frame.from_options(
    frame=final_df,
    connection_type="s3",
    format="csv",
    connection_options={"path": "s3://credit-transaction-fruad-new-processed", "partitionKeys": []},
)

job.commit()
