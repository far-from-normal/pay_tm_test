# %% [markdown]
# # Paytm test

# %% [markdown]
# ## Python imports
# All imports have been moved to this location to avoid clutter further down in the code.

'''
### Step 1 - Setting Up the Data

1. Load the global weather data into your big data technology of choice.
2. Join the stationlist.csv with the countrylist.csv to get the full country name for each station number.
3. Join the global weather data with the full country names by station number.

We can now begin to answer the weather questions! 

### Step 2 - Questions
Using the global weather data, answer the following:

1. Which country had the hottest average mean temperature over the year?
2. Which country had the most consecutive days of tornadoes/funnel cloud formations?
3. Which country had the second highest average mean wind speed over the year?

## What are we looking for?
We want to see how you handle:

* Code quality and best practices
* New technologies and frameworks
* Messy (ie real) data
* Understanding of data transformation. This is not a pass or fail test, we want to hear about your challenges and your successes with this challenge.
'''


# %% Module Imports

# pyspark imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
from pyspark.sql import functions as F
from pyspark.sql.window import Window as W

# %% Define Spark Session
spark = (
    SparkSession
    .builder
    .master("local[*]")
    .appName("sobeys_unit_forecaster")
    .config("spark.sql.shuffle.partitions", 500)
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)


# %% [markdown]
# ## Step 1 - Setting Up the Data

# %% 1. Load the global weather data into your big data technology of choice.

df_s = (
    spark.read.options(header=True, delimiter=',')
    # .schema(schema_s)
    .csv("stationlist.csv")
)

df_c = (
    spark.read.options(header=True, delimiter=',')
    # .schema(schema_s)
    .csv("countrylist.csv")
)

# main table schema
fields_19 = [
    StructField("STN---", IntegerType(), True), 
    StructField("WBAN", IntegerType(), True), 
    StructField("YEARMODA", IntegerType(), True),
    StructField("TEMP", DoubleType(), True),
    StructField("DEWP", DoubleType(), True),
    StructField("SLP", DoubleType(), True),
    StructField("VISIB", DoubleType(), False),
    StructField("WDSP", DoubleType(), True),
    StructField("MXSPD", DoubleType(), False),
    StructField("GUST", DoubleType(), False),
    StructField("MAX", DoubleType(), False),
    StructField("MIN", DoubleType(), False),
    StructField("PRCP", DoubleType(), True),
    StructField("SNDP", DoubleType(), True),
    StructField("FRSHTT", StringType(), True),
]
schema_19 = StructType(fields=fields_19)


df = (
    spark.read.options(header=True, delimiter=',', inferSchema=True)
    # .schema(schema_19)
    .csv("data/2019/*")
)


print(df_s.printSchema())
print(df_s.show())

print(df_c.printSchema())
print(df_c.show())

print(df.printSchema())
print(df.show())

# %% 2. Join the stationlist.csv with the countrylist.csv to get the full country name for each station number.
# left join is used here in case the countryList table does not contain some COUNTRY_ABBR values in stationlist

df_sc = (
    df_s
    .join(
        df_c,
        on=[
            "COUNTRY_ABBR"
        ],
        how="left"
    )
)

print(df_sc.printSchema())
print(df_sc.show())


# %% 3. Join the global weather data with the full country names by station number.
# Using another left join in case some "STN_NO" values are not contained in /data/2019/*
# renameing

df = (
    df
    .withColumnRenamed("STN---", "STN_NO")
    .join(
        df_sc,
        on=[
            "STN_NO"
        ],
        how="left"
    )
)

print(df.printSchema())
print(df.show())

df = (
    df
    .withColumn("FRSHTT", F.col("FRSHTT").cast(StringType()))
)

# Fields MAX, MIN, PRCP are inferred as StringType. Could strip columns for alpha characters here, then cast to DoubleType

# %% Dealing with missing values

df = (
    df
    .withColumn("TEMP", F.when(~F.col("TEMP").isin(9999.9), F.col("TEMP")))
    .withColumn("DEWP", F.when(~F.col("DEWP").isin(9999.9), F.col("DEWP")))
    .withColumn("SLP", F.when(~F.col("SLP").isin(9999.9), F.col("SLP")))
    .withColumn("STP", F.when(~F.col("STP").isin(9999.9), F.col("STP")))
    .withColumn("VISIB", F.when(~F.col("VISIB").isin(999.9), F.col("VISIB")))
    .withColumn("WDSP", F.when(~F.col("WDSP").isin(999.9), F.col("WDSP")))
    .withColumn("MXSPD", F.when(~F.col("MXSPD").isin(999.9), F.col("MXSPD")))
    .withColumn("GUST", F.when(~F.col("GUST").isin(999.9), F.col("GUST")))
    .withColumn("MAX", F.when(~F.col("MAX").isin('9999.9'), F.col("MAX")))
    .withColumn("MIN", F.when(~F.col("MIN").isin('9999.9'), F.col("MIN")))
    .withColumn("PRCP", F.when(~F.col("PRCP").isin('99.99'), F.col("PRCP")))
    .withColumn("SNDP", F.when(~F.col("SNDP").isin(999.9), F.col("SNDP")))
)

print(df.printSchema())
print(df.show(100))


# %% 1. Which country had the hottest average mean temperature over the year?

df_hamt = (
    df
    .groupBy(["COUNTRY_FULL"])
    .agg(F.mean("TEMP").alias("AVG_MEAN_TEMP"))
    .orderBy(F.col("AVG_MEAN_TEMP").desc())
    .limit(1)
    .select("COUNTRY_FULL")
)

print("The country that had the hottest average mean temperature over the year is ", df_hamt.show())
# DJIBOUTI

# %% 2. Which country had the most consecutive days of tornadoes/funnel cloud formations?

df_tfct = (
    df
    .withColumn("TORNADOS", F.substring(F.col("FRSHTT"), 6, 1))
    .groupBy(["COUNTRY_FULL", "YEARMODA"])
    .agg(F.max("TORNADOS").alias("TORNADOS"))
    .filter(F.col("TORNADOS") == 1)
    .withColumn("DATECOL", F.unix_timestamp(F.col("YEARMODA").cast(StringType()), "yyyyMMdd").cast("timestamp"))
    .withColumn("DATEDIFF_TORNADO", 
        F.datediff(
            F.col("DATECOL"), 
            F.lag(F.col("DATECOL"), 1).over(W.partitionBy("COUNTRY_FULL").orderBy("DATECOL"))
            )
    )
    .groupBy(["COUNTRY_FULL"])
    .agg(F.max("DATEDIFF_TORNADO").alias("CONSECUTIVE_DAYS_TORNADOS"))
    .orderBy(F.col("CONSECUTIVE_DAYS_TORNADOS").desc())
    .limit(1)
    .select("COUNTRY_FULL")
)

print("The country that had the most consecutive days of tornadoes/funnel cloud formations is ", df_tfct.show())
# CUBA

# %% 3. Which country had the second highest average mean wind speed over the year?

df_shws = (
    df
    .groupBy(["COUNTRY_FULL"])
    .agg(F.mean("WDSP").alias("AVG_MEAN_WDSP"))
    .withColumn("rank", F.rank().over(W.orderBy(F.col("AVG_MEAN_WDSP").desc())))
    .filter(F.col("rank") == 2)
    .select("COUNTRY_FULL")
)

print("The country that had the second highest average mean wind speed over the year is ", df_shws.show())
# ARUBA