// Databricks notebook source
// Moving the zip file to LFS (Local File system) to unzip
dbutils.fs.mv( "dbfs:/FileStore/tables/train.zip", "file:/databricks/driver/")

// COMMAND ----------

// MAGIC %sh
// MAGIC unzip train.zip

// COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/train.csv", "dbfs:/tmp/train.csv")

// COMMAND ----------

// MAGIC %sh
// MAGIC rm train.zip

// COMMAND ----------

// MAGIC %python
// MAGIC input_df = spark.read.csv("/tmp/train.csv", header="true", inferSchema="true")

// COMMAND ----------

// MAGIC %python
// MAGIC input_df.repartition(10).write.mode('overwrite').parquet("/tmp/train.parquet")

// COMMAND ----------

// MAGIC %python
// MAGIC df = spark.read.parquet("/tmp/train.parquet")

// COMMAND ----------

// MAGIC %python
// MAGIC df.describe().show()

// COMMAND ----------

// MAGIC %python
// MAGIC df.count()

// COMMAND ----------

// MAGIC %python
// MAGIC df_without_nulls=df.na.drop()
// MAGIC df_without_nulls.limit(2000000).createOrReplaceTempView("nyc_taxi_table")

// COMMAND ----------

// MAGIC %python
// MAGIC df_without_nulls.count()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Tidying the dataset by removing outliers

// COMMAND ----------

// MAGIC %python
// MAGIC df_tidy = spark.sql("SELECT * FROM nyc_taxi_table \
// MAGIC WHERE  \
// MAGIC (fare_amount > 0 AND fare_amount <= (SELECT AVG(fare_amount) + 3*stddev(fare_amount) FROM nyc_taxi_table)) \
// MAGIC AND (passenger_count > 0 AND passenger_count <= (SELECT ROUND(AVG(passenger_count) + 3*stddev(fare_amount)) FROM nyc_taxi_table)) \
// MAGIC AND (pickup_longitude BETWEEN (SELECT AVG(pickup_longitude) - 3*stddev(pickup_longitude) FROM nyc_taxi_table) AND (SELECT AVG(pickup_longitude) + 3*stddev(pickup_longitude) FROM nyc_taxi_table))  \
// MAGIC AND (pickup_latitude BETWEEN (SELECT AVG(pickup_latitude) - 3*stddev(pickup_latitude) FROM nyc_taxi_table) AND (SELECT AVG(pickup_latitude) + 3*stddev(pickup_latitude) FROM nyc_taxi_table))  \
// MAGIC AND (pickup_latitude BETWEEN (SELECT AVG(pickup_latitude) - 3*stddev(pickup_latitude) FROM nyc_taxi_table) AND (SELECT AVG(pickup_latitude) + 3*stddev(pickup_latitude) FROM nyc_taxi_table)) \
// MAGIC AND (dropoff_longitude BETWEEN (SELECT AVG(dropoff_longitude) - 3*stddev(dropoff_longitude) FROM nyc_taxi_table) AND (SELECT AVG(dropoff_longitude) + 3*stddev(dropoff_longitude) FROM nyc_taxi_table))\
// MAGIC AND (dropoff_latitude BETWEEN (SELECT AVG(dropoff_latitude) - 3*stddev(dropoff_latitude) FROM nyc_taxi_table) AND (SELECT AVG(dropoff_latitude) + 3*stddev(dropoff_latitude) FROM nyc_taxi_table))")

// COMMAND ----------

// MAGIC %python
// MAGIC df_tidy.cache()
// MAGIC df_tidy.count()

// COMMAND ----------

dbutils.library.installPyPI("geopandas", version="0.5.1")

// COMMAND ----------

// MAGIC %python
// MAGIC from geopy.geocoders import Nominatim
// MAGIC from pyspark.sql.functions import lit, struct
// MAGIC geolocator = Nominatim(user_agent="agentpre")
// MAGIC #location = geolocator.reverse("52.509669, 13.376294")
// MAGIC 
// MAGIC from pyspark.sql.functions import udf
// MAGIC from geopy.distance import geodesic
// MAGIC 
// MAGIC @udf("float")
// MAGIC def geodesic_udf(x, y):
// MAGIC     return geodesic(x, y).miles
// MAGIC 
// MAGIC 
// MAGIC @udf("string")
// MAGIC def address_finder(a):
// MAGIC   s=geolocator.reverse(a)
// MAGIC   return s.address
// MAGIC 
// MAGIC from pyspark.sql.functions import *
// MAGIC df_tidy_expanded=df_tidy.select(col("*"),year('key').alias('year'),month('key').alias('month'),dayofmonth('key').alias('day'), hour('key').alias('hour'), minute('key').alias('minute'), second('key').alias('second'), dayofweek('key').alias('day_of_week') ).withColumn("distance", geodesic_udf(struct(col("pickup_latitude"),col("pickup_longitude")), struct(col("dropoff_latitude"),col("dropoff_longitude")))).withColumn("fare_per_mile", col("fare_amount")/col("distance"))
// MAGIC #.orderBy(rand())
// MAGIC #.withColumn("pickup_address", address_finder(concat(col("pickup_latitude"), lit(","),col("pickup_longitude") ))).withColumn("pickup_address", address_finder(concat(col("pickup_latitude"), lit(","),col("pickup_longitude") )))
// MAGIC 
// MAGIC #.orderBy(rand()).limit(10000)

// COMMAND ----------

// MAGIC %python
// MAGIC #df_tidy_expanded.orderBy(rand()).limit(5400000).coalesce(8).write.mode('overwrite').parquet("/tmp/expanded.parquet")
// MAGIC df_tidy_expanded.coalesce(8).write.mode('overwrite').parquet("/tmp/expanded1.parquet")

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql.functions import *
// MAGIC 
// MAGIC df_tidy_expanded_new = spark.read.parquet("/tmp/expanded1.parquet")
// MAGIC 
// MAGIC from pyspark.sql.functions import *
// MAGIC df_tidy_expanded_new1=df_tidy_expanded_new.select(col("*"), date_format("key", "E").alias('daynameoftheweek'), date_format("key", "MMM").alias('nameofmonth') )
// MAGIC #.withColumn("distance", geodesic_udf(struct(col("pickup_latitude"),col("pickup_longitude")), struct(col("dropoff_latitude"),col("dropoff_longitude")))).withColumn("fare_per_mile", col("fare_amount")/col("distance"))
// MAGIC 
// MAGIC #df_tidy_expanded_new1=df_tidy_expanded_new.select(col("*"),date_format(col("pickup_datetime", "E").alias('daynameoftheweek'))
// MAGIC 
// MAGIC #test_locations=df_tidy_expanded.orderBy(rand()).limit(1000).cache()
// MAGIC df_tidy_expanded_new1.cache()
// MAGIC df_tidy_expanded_new1.createOrReplaceTempView("df_tidy_expanded")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT  dropoff_latitude, dropoff_longitude, dropoff_count, pas, fare  FROM (SELECT  dropoff_latitude, dropoff_longitude, COUNT(key) AS dropoff_count, AVG(passenger_count) AS pas, AVG(fare_amount) AS fare
// MAGIC FROM df_tidy_expanded 
// MAGIC GROUP BY dropoff_latitude, dropoff_longitude HAVING  (dropoff_latitude BETWEEN  40.495992 AND 40.915568) AND (dropoff_longitude BETWEEN -74.257159 AND -73.699215))
// MAGIC ORDER BY dropoff_count DESC
// MAGIC LIMIT 100

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT  pickup_latitude, pickup_longitude, pickup_count, avg_passengers, fare_avg, distance_avg  FROM (SELECT  pickup_latitude, pickup_longitude, COUNT(key) AS pickup_count, AVG(passenger_count) AS avg_passengers, AVG(fare_amount) AS fare_avg, AVG(distance) AS distance_avg
// MAGIC FROM df_tidy_expanded 
// MAGIC GROUP BY pickup_latitude, pickup_longitude HAVING  (pickup_latitude BETWEEN  40.495992 AND 40.915568) AND (pickup_longitude BETWEEN -74.257159 AND -73.699215))
// MAGIC ORDER BY pickup_count DESC
// MAGIC LIMIT 100

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC pickup_address = spark.sql("SELECT  pickup_latitude, pickup_longitude, pickup_count, avg_passengers, fare_avg, distance_avg  FROM (SELECT  pickup_latitude, pickup_longitude, COUNT(key) AS pickup_count, AVG(passenger_count) AS avg_passengers, AVG(fare_amount) AS fare_avg, AVG(distance) AS distance_avg \
// MAGIC FROM df_tidy_expanded \
// MAGIC GROUP BY pickup_latitude, pickup_longitude HAVING  (pickup_latitude BETWEEN  40.495992 AND 40.915568) AND (pickup_longitude BETWEEN -74.257159 AND -73.699215)) \
// MAGIC ORDER BY pickup_count DESC \
// MAGIC LIMIT 100").withColumn("pickup_address",  address_finder(concat(col("pickup_latitude"), lit(","),col("pickup_longitude" ))))
// MAGIC 
// MAGIC pickup_address.show(10)
// MAGIC pickup_address.cache()

// COMMAND ----------

// MAGIC %python
// MAGIC pickup_address.show(10)

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT nameofmonth FROM test_locations  LIMIT 1

// COMMAND ----------

// MAGIC %md
// MAGIC ## Business Problem:
// MAGIC As the taxi market is contracting, it is important to identify the best opportunities for the taxi drivers to gain revenue. This analysis aims at helping taxi drivers in identifying the best time of the day, best day of the week and best pickup location for the longest or the shortest ride.
// MAGIC 
// MAGIC From 2009 to mid 2012, the average taxi fare per year varies only by about $1 range, with a slight peak in May/June. Except in Sept 2012, there is a 18% increase from $10.59 to $12.46 within a month. It can be explained by the fact that there is a vote for 17% fare increase approved by The Taxi and Limousine Commission and is taken into effect in Sept 2012. After the increase, the average fare continues to vary within the $1 range.

// COMMAND ----------

// MAGIC %md # 18% increase in average fare per trip in September 2012

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC -- Trend of fare by month and by year
// MAGIC SELECT nameofmonth AS Month, month, avg(fare_amount) AS Fare, year AS Year
// MAGIC FROM df_tidy_expanded
// MAGIC WHERE NOT (Year=2015)
// MAGIC GROUP BY nameofmonth, month, Year
// MAGIC ORDER BY month, Year

// COMMAND ----------

// MAGIC %md 
// MAGIC The official taxi fare increase does not the taxi driver for long.  Fewer people are taking taxi in New York City starting in 2012.  The total number of taxi rides in 2012 drops from 8.62m to 8.46m in 2013, and continues to drop to 8.05m in 2014.  In 2014, we start to see a decrease in the total taxi fare, representing a revenue drop of 3% in one year.

// COMMAND ----------

// MAGIC %md # Number of taxi trips continues to drop from 2012

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC -- Trend of fare by month and by year
// MAGIC SELECT nameofmonth AS Month, month, count(*) AS Count, year AS Year
// MAGIC FROM df_tidy_expanded
// MAGIC WHERE NOT (Year=2015)
// MAGIC GROUP BY nameofmonth, month, Year
// MAGIC ORDER BY month, Year

// COMMAND ----------

// MAGIC %md # Taxi revenue starts decreasing in 2013

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC -- Trend of fare by month and by year
// MAGIC SELECT nameofmonth AS Month, month, sum(fare_amount) AS Total_Fare, year AS Year
// MAGIC FROM df_tidy_expanded
// MAGIC WHERE NOT (Year=2015)
// MAGIC GROUP BY nameofmonth, month, Year
// MAGIC ORDER BY month, Year

// COMMAND ----------

// MAGIC %md 
// MAGIC # Most taxi riders on Fri.  
// MAGIC # Least taxi riders on Mon.

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC -- Trend of trips by day of the week
// MAGIC SELECT daynameoftheweek AS Day_of_Week, month, year AS Year, count(*) AS Number_of_Trips, day_of_week dayoftheweek
// MAGIC FROM df_tidy_expanded
// MAGIC WHERE NOT (Year=2015)
// MAGIC GROUP BY dayoftheweek, daynameoftheweek, month, Year
// MAGIC ORDER BY dayoftheweek, daynameoftheweek, month, Year

// COMMAND ----------

// MAGIC %md 
// MAGIC # Peak hours between 18:00 and 22:00.  
// MAGIC # Off-peak hours between 01:00 and 06:00.

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Trend of trips by hour of the day
// MAGIC SELECT hour AS Hour, count(*) AS Trips, year AS Year
// MAGIC FROM df_tidy_expanded
// MAGIC WHERE Year <> 2015
// MAGIC GROUP BY Hour, Year
// MAGIC ORDER BY Hour, Year

// COMMAND ----------

// MAGIC %md
// MAGIC The average fare is the highest at 5am.
// MAGIC Given that there are the least number of taxi trips at 5am, the highest fare suggests that the trips are long rides.

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Trend of fare by hour of the day
// MAGIC 
// MAGIC SELECT sum(fare_amount)/count(*) AS Fare, hour AS Hour
// MAGIC FROM df_tidy_expanded
// MAGIC GROUP BY Hour
// MAGIC ORDER BY Hour

// COMMAND ----------

// MAGIC %md
// MAGIC #### Question No: 6
// MAGIC ##### What is the average fare/mile during each day of the week?

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC SELECT day_of_week , AVG(fare_per_mile) AVG_Fare_Per_Mile
// MAGIC FROM df_tidy_expanded
// MAGIC GROUP BY day_of_week
// MAGIC ORDER BY day_of_week;

// COMMAND ----------

// MAGIC %python
// MAGIC import pandas as pd
// MAGIC import numpy as np
// MAGIC import matplotlib.pyplot as plt
// MAGIC import datetime

// COMMAND ----------



// COMMAND ----------




// COMMAND ----------

// MAGIC %md
// MAGIC ### Question No: 4
// MAGIC #### Is there any difference in fare prices druing weekdays and weekends?

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC SELECT we.hour, Weekend_Fare , Weekday_Fare
// MAGIC FROM (
// MAGIC SELECT hour, AVG(fare_amount) Weekend_Fare
// MAGIC FROM df_tidy_expanded
// MAGIC WHERE day_of_week IN (6,7)
// MAGIC GROUP BY hour) we
// MAGIC INNER JOIN (
// MAGIC SELECT hour, AVG(fare_amount) Weekday_Fare
// MAGIC FROM df_tidy_expanded
// MAGIC WHERE day_of_week BETWEEN 1 AND 5
// MAGIC GROUP BY hour) wd
// MAGIC ON we.hour = wd.hour
// MAGIC ORDER BY hour
// MAGIC ;

// COMMAND ----------

// MAGIC %md
// MAGIC ### Question No:5 
// MAGIC #### What is the average fare amount over years during the rush/peak times?

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT year, AVG(fare_amount) AS Fare_Amount
// MAGIC FROM df_tidy_expanded
// MAGIC WHERE (hour >= 16 and hour <=20) AND (day_of_week >=1 and day_of_week <=5)
// MAGIC GROUP BY year
// MAGIC ORDER BY year;

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT COUNT(*) FROM df_tidy_expanded
