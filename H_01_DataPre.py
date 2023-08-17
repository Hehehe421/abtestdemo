# Databricks notebook source
# MAGIC %run ./00_setup

# COMMAND ----------

from pyspark.sql.functions import col, lit, rand, when
from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np
import datetime
from random import choice, randint
from pyspark.sql import functions as F
from random import choice, randint, uniform
from datetime import datetime, timedelta

# COMMAND ----------

chkpt_path = "/tmp/summer/test_chkpts/"
df = (spark.read.format("delta")
      .option("cloudFiles.format", "json")
      .option("cloudFiles.schemaEvolutionMode","addNewColumns")
      .option("cloudFiles.schemaLocation", chkpt_path)
      .load(input_path))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T

@F.udf(returnType=T.StringType())
def get_string_to_length(text):
    nulls_to_fill = 8 - len(text) 
    return '0' * nulls_to_fill+text    

# Add the fake user_id for the use case
df = df.withColumn('listing_id', get_string_to_length(F.col('listing_id')))
# Reorder the columns
df_selected = df.select(["listing_id", "price", "neighbourhood_cleansed", "room_type", "instant_bookable", "accommodates", "bedrooms", "beds", "minimum_nights", "maximum_nights"])

# COMMAND ----------

df_selected.display()

# COMMAND ----------

# Delete the old database and tables if needed (for demo purposes)
spark.sql(f'DROP DATABASE IF EXISTS {database_name} CASCADE')
# Create database to house tables
spark.sql(f'CREATE DATABASE {database_name}')

# Create a Delta Lake table from loaded 
#bronze_tbl_name = 'listing_bronze'
#df_selected.write.format('delta').mode('overwrite').saveAsTable(f'{database_name}.{bronze_tbl_name}')


# COMMAND ----------

# MAGIC %md
# MAGIC Make up the user activities

# COMMAND ----------

user_df = (spark.read.csv('/dbdemos/fsi/fraud-detection/customers', header=True, multiLine=True))

# COMMAND ----------

# Add the fake user_id for the use case
from pyspark.sql.functions import monotonically_increasing_id, col

user_df = user_df.withColumn('user_id', monotonically_increasing_id()).filter(col("user_id") <= 64940) 
#user_df = user_df.withColumn('user_id', get_string_to_length(F.col('user_id').cast("string")))

# Add is_host column with random values
user_df = user_df.withColumn("is_host", F.expr("boolean(RAND() <= 0.2)"))

user_df_selected = user_df.select(["user_id", "is_host","firstname", "lastname", "email", "country", "creation_date"])

# COMMAND ----------

user_df_selected.display()

# COMMAND ----------

# Filter out is_host = true and assigne back to the listing table
host_users_df = user_df_selected.filter(F.col("is_host") == True)
host_user_ids = host_users_df.select("user_id").rdd.flatMap(lambda x: x).collect()

# Function to assign a random host user ID
def assign_random_host_user_id():
    return choice(host_user_ids)

# Register the UDF
spark.udf.register("assign_random_host_user_id", assign_random_host_user_id)

# Add a new column with randomly assigned host user IDs
listing_df_with_user_id = df_selected.withColumn("user_id", F.expr("assign_random_host_user_id()"))

# Add a new column with randomly assigned review points between 10 and 100
listing_df_with_user_id = listing_df_with_user_id.withColumn("review_score", F.expr("int(RAND() * 91) + 10"))


# COMMAND ----------

listing_df_with_user_id.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Make up the user activities daily log, here is the logic to generate the dataset:
# MAGIC - Define a function called generate_activity_data that takes user_ids, listing_ids, start_date, and end_date as inputs.
# MAGIC - Create an empty list called data to store the simulated activity data.
# MAGIC - Run a loop from start_date to end_date.
# MAGIC - Inside the loop, generate a random number of users (between 50 and 300) for the current date.
# MAGIC - Randomly select a subset of user_ids for the current date.
# MAGIC - - For each selected user, generate a random number of listings (between 1 and 20).
# MAGIC - Randomly select a subset of listing_ids for each user.
# MAGIC - For each selected user and listing pair, generate an event_time by adding a random number of seconds to the start_date.
# MAGIC - Append a tuple (user_id, listing_id, event_time, "view", None) to the data list to simulate a "view" activity.
# MAGIC - Based on random probabilities, append additional tuples to simulate "click," "save," or "purchase" activities with corresponding activity types and purchase amounts.
# MAGIC - Increment the start_date by one day in each iteration.

# COMMAND ----------

from pyspark.sql.functions import col
from datetime import datetime, timedelta
import random

# Simulate user activities
def generate_activity_data(user_ids, listing_ids, start_date, end_date):
    data = []
    
    while start_date <= end_date:
        num_users = random.randint(50, 300)
        selected_user_ids = random.sample(user_ids, num_users)
        
        for user_id in selected_user_ids:
            num_listings = random.randint(1, 20)
            selected_listing_ids = random.sample(listing_ids, num_listings)
            
            for listing_id in selected_listing_ids:
                event_time = start_date + timedelta(seconds=random.randint(0, 86400))
                data.append((user_id, listing_id, event_time, "view", None))
                
                if random.random() < 0.3:
                    data.append((user_id, listing_id, event_time, "click", None))
                if random.random() < 0.15:
                    data.append((user_id, listing_id, event_time, "save", None))
                if random.random() < 0.1:
                    purchase_amount = round(random.uniform(200, 500), 2)
                    data.append((user_id, listing_id, event_time, "purchase", purchase_amount))
        
        start_date += timedelta(days=1)
    
    return data

# Define schema for the DataFrame
schema = ["user_id", "listing_id", "event_time", "activity", "purchase_amount"]

# Generate user and listing IDs
# Test ids
#user_ids = list(range(1, 1001))  # Example: 1000 user IDs
#listing_ids = list(range(1, 501))  # Example: 500 listing IDs

user_ids = user_df_selected.select("user_id").rdd.flatMap(lambda x: x).collect()
listing_ids = df_selected.select('listing_id').distinct().rdd.map(lambda x: x[0]).collect()

# Generate simulation data
start_date = datetime(2023, 7, 1)
end_date = datetime(2023, 7, 31)
simulation_data = generate_activity_data(user_ids, listing_ids, start_date, end_date)

# Create DataFrame from simulation data
activity_df = spark.createDataFrame(simulation_data, schema)


# COMMAND ----------

activity_df.display()

# COMMAND ----------

activity_df.describe().show()

# COMMAND ----------

# MAGIC %md save the data into Delta

# COMMAND ----------

# Create a Delta Lake table from loaded 
# listing data
bronze_tbl_name = 'listing_bronze'
listing_df_with_user_id.write.format('delta').mode('overwrite').saveAsTable(f'{database_name}.{bronze_tbl_name}')
# user data
bronze_tbl_name = 'user_bronze'
user_df_selected.write.format('delta').mode('overwrite').saveAsTable(f'{database_name}.{bronze_tbl_name}')
# log data
userlog_tbl_name = 'userlog_bronze'
activity_df.write.format('delta').mode('overwrite').saveAsTable(f'{database_name}.{userlog_tbl_name}')


# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES from abtest_workshop_sixuan_he;

# COMMAND ----------


