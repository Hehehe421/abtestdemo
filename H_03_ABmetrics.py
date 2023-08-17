# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### Overall Evaluation Criterion or OEC
# MAGIC OEC or Overall Evaluation Criterion is the most vital metric during experimentation. It is a unit of measurement that evaluates user satisfaction and long-term business value. Here is an example; Netflix is a subscription-based business. Its OEC will be the viewing hours of a customer. If one user watches Netflix for one hour a month and the other for 15 hours. Then the second user is likely to renew the monthly subscription. Viewing hours are the key metric, also termed retention, which indicates the number of users who return every month. As evident, viewing hours or content consumption becomes the key proxy metric or the OEC. 
# MAGIC
# MAGIC Here, let's assume we have 2 OEC:
# MAGIC - **Active Rate**: Defined as total active users / total users per day, week, monthly
# MAGIC - **Average Duration**: Defined as average duration engaged on the website per user per day, week, monthly

# COMMAND ----------

# MAGIC %md
# MAGIC ## Common Types of Statistical Test
# MAGIC By employing these statistical tests in AB testing scenarios, we can effectively analyze the data and draw conclusions regarding the significance and impact of different interventions or treatments. Here are some common types of statistical testing:
# MAGIC
# MAGIC - **Ratio Chi-square Test**: The Ratio Chi-square Test is employed to examine the equality of proportions between two or more groups. It is particularly useful in AB testing scenarios to ensure a fair and unbiased split between the control and treatment groups. The test calculates the chi-square statistic, which is then compared to the critical chi-square value to determine statistical significance.
# MAGIC
# MAGIC - **Proportion Z-test**: The Proportion Z-test is utilized to determine whether there is a significant difference in proportions between two groups. It is commonly employed in AB testing to compare the success rates, conversion rates, or engagement rates of different interventions. The test calculates the Z-score, which is then compared to the critical Z-value to assess statistical significance.
# MAGIC
# MAGIC - **Two Sample Mean T-test**: The Two Sample Mean T-test is employed to determine if there is a significant difference between the means of two independent groups. It is frequently used in AB testing to compare the average values of a specific metric between the control and treatment groups. The test calculates the T-value, which is then compared to the critical T-value to assess statistical significance.
# MAGIC

# COMMAND ----------

def display_slide(slide_id, slide_number):
  displayHTML(f'''
  <div style="width:1150px; margin:auto">
  <iframe
    src="https://docs.google.com/presentation/d/{slide_id}/embed?slide={slide_number}"
    frameborder="0"
    width="1150"
    height="683"
  ></iframe></div>
  ''')

display_slide('1c-uVbdQFSBYIadc4NcI56_o1JPMYAWpOeOTGqOTph4M', 15)  #hide this code

# COMMAND ----------

display_slide('1c-uVbdQFSBYIadc4NcI56_o1JPMYAWpOeOTGqOTph4M', 16)  #hide this code

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import lit
import random
import pandas as pd
import numpy as np
import datetime
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
from pyspark.sql.functions import count, lit, to_date, date_trunc, date_add,countDistinct
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from scipy.stats import chisquare
from pyspark.sql.functions import rand, expr, col, when
from pyspark.sql.functions import weekofyear, date_trunc, sum, when
from scipy.stats import ttest_ind
from pyspark.sql.functions import col, sum
from scipy.stats import ttest_ind
from pyspark.sql.window import Window
from pyspark.sql.functions import sum as sql_sum

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC SHOW TABLES from abtest_workshop_sixuan_he;

# COMMAND ----------

silver_tbl_name = 'user_silver'
user_experiment = (spark.table(f'{database_name}.{silver_tbl_name}'))

# COMMAND ----------

user_experiment.display()

# COMMAND ----------

userlogbronze_tbl_name = 'userlog_bronze'
user_activity_df = (spark.table(f'{database_name}.{userlogbronze_tbl_name}'))
spark.sql(f'SELECT * FROM {database_name}.{userlogbronze_tbl_name}').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ratio Chi-squre Test
# MAGIC
# MAGIC This test compares the observed group sizes to the expected group sizes under the null hypothesis that the groups are split as expected:
# MAGIC
# MAGIC - The null hypothesis (H0) for the test is that the ratio of control and treamtment are the same as expected equal = 0.5.
# MAGIC - The alternate hypothesis (H1) is that the ratio of control and treamtment are the **not** same as expected equal = 0.5.

# COMMAND ----------

import matplotlib.pyplot as plt

# group the DataFrame by the "group" column and count the number of rows in each group
group_counts = user_experiment.groupBy("group").count().collect()

# extract the group labels and counts from the resulting DataFrame
labels = [row["group"] for row in group_counts]
counts = [row["count"] for row in group_counts]

# plot the bar chart
plt.bar(labels, counts)
plt.title("Group Counts")
plt.xlabel("Group")
plt.ylabel("Count")
plt.show()

# COMMAND ----------

from builtins import round
def ratio_test(user_df, group, ratio, experiment_id):
    user_df = user_df.filter(col("experiment_id") == experiment_id)
    total_count = user_df.count()
    treatment_count = int(total_count * ratio)
    control_count = total_count - treatment_count
    expected = [control_count, treatment_count]
    observed = user_df.groupBy(group).count().orderBy(group).select('count').rdd.flatMap(lambda x: x).collect()
    if len(observed) != 2:
        raise ValueError("Expected 2 groups, but found {}".format(len(observed)))
    control_observed = int(observed[0])
    treatment_observed = int(observed[1])
    control_expected = int(expected[0])
    treatment_expected = int(expected[1])
    chi_sq, p_value = chisquare([control_observed, treatment_observed], f_exp=[control_expected, treatment_expected])
    test_stat = round(float(chi_sq), 6)
    p_value = round(float(p_value), 6)
    #return chi_sq, p_value
    return spark.createDataFrame([(experiment_id, control_observed, control_expected, treatment_observed, treatment_expected, test_stat, p_value)], 
                                 ['experiment_id', 'control_observed', 'control_expected', 'treatment_observed', 'treatment_expected', 'test_stat', 'p_value'])


# COMMAND ----------

result_df = ratio_test(user_experiment, "group", 0.5, 'experiment_20230817_000145_435405')

# COMMAND ----------

ratio_test = 'ratio_test'
# Create a Delta Lake table from loaded 
result_df.write.format('delta').mode('overwrite').saveAsTable(f'{database_name}.{ratio_test}')

# COMMAND ----------

result_df = ratio_test(user_experiment, "group", 0.2, 'experiment_20230817_162908_575997')

# COMMAND ----------

ratio_test = 'ratio_test'
# Create a Delta Lake table from loaded 
result_df.write.format('delta').mode('append').saveAsTable(f'{database_name}.{ratio_test}')

# COMMAND ----------

# MAGIC %md
# MAGIC Nice, we have p_value > 0.05 to accept our null hypothesis, so that our AB experiment is split as expected

# COMMAND ----------

# MAGIC %md
# MAGIC ### Z proportion test -- Active Rate 
# MAGIC
# MAGIC This tests for a difference in proportions. A two proportion z-test allows you to compare two proportions to see if they are the same.
# MAGIC
# MAGIC - The null hypothesis (H0) for the test is that the proportions are the same.
# MAGIC - The alternate hypothesis (H1) is that the proportions are not the same.
# MAGIC
# MAGIC In this demo, we would like to test whether the Active Rate are statically different between the control group and the treatment group.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Average Total Purchase Two sample mean T-test
# MAGIC
# MAGIC The two-sample t-test (Snedecor and Cochran, 1989) is used to determine if two population means are equal. In this demo, we would like to test the average total duration per customer between the control and treatment group.
# MAGIC - The null hypothesis (H0) for the test is that the proportions are the same.
# MAGIC - The alternate hypothesis (H1) is that the proportions are not the same.

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES from abtest_workshop_sixuan_he;

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

user_activity_df.display()

# COMMAND ----------

# Define the start and end dates
start_date = '2023-07-01'
end_date = '2023-07-31'

# Create a list of dates between the start and end dates
dates = [str(date.date()) for date in pd.date_range(start=start_date, end=end_date)]

# Create a PySpark DataFrame with the list of dates
date_df = spark.createDataFrame(pd.DataFrame({'date': dates}))

# Add columns for week start and end dates
date_df = date_df.withColumn('date', to_date('date')) \
                 .withColumn('week_start_date', date_trunc('week', 'date')) \
                 .withColumn('week_end_date', date_add('week_start_date', 6))

# COMMAND ----------

# Read the userlog_bronze data
userlogbronze_tbl_name = 'userlog_bronze'
master_df = (spark.table(f'{database_name}.{userlogbronze_tbl_name}'))
master_df.display()

# COMMAND ----------

user_experiment.display()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Replace null values in purchase_amount with zeros
#user_log_df = user_log_df.withColumn("purchase_amount", F.coalesce("purchase_amount", F.lit(0.0)))
user_activity_df = user_activity_df.withColumn("purchase_amount", F.coalesce("purchase_amount", F.lit(0.0)))
# Define aggregation window specification
window_spec = F.window("event_time", "1 day")

# Define aggregation expressions
aggregation_exprs = [
    F.date_format(window_spec.start, "yyyy-MM-dd").alias("date"),
    F.countDistinct("listing_id").alias("total_listings"),
    F.sum(F.when(F.col("activity") == "view", 1).otherwise(0)).alias("total_views"),
    F.sum(F.when(F.col("activity") == "click", 1).otherwise(0)).alias("total_clicks"),
    F.sum(F.when(F.col("activity") == "save", 1).otherwise(0)).alias("total_saves"),
    F.sum(F.when(F.col("activity") == "comment", 1).otherwise(0)).alias("total_comments"),
    F.sum(F.when(F.col("activity") == "purchase", 1).otherwise(0)).alias("total_purchases"),
    F.sum(F.when(F.col("activity") == "purchase", F.col("purchase_amount")).otherwise(0)).alias("total_purchase_amount")
]

# Aggregate data
aggregated_user_log_df = user_activity_df.groupBy("user_id", window_spec).agg(*aggregation_exprs)


# COMMAND ----------

aggregated_user_log_df.display()

# COMMAND ----------

experiment_1 = user_experiment.filter(col("experiment_id") == 'experiment_20230817_000145_435405')

# COMMAND ----------

experiment_1.display()

# COMMAND ----------


aggregated_user_log_df2 = aggregated_user_log_df.alias('ua').join(experiment_1.alias('wa'), 
                                                     (col('ua.user_id') == col('wa.user_id')), 
                                                     how='inner')



# COMMAND ----------

aggregated_user_log_df2.display()

# COMMAND ----------

adjusted_df = aggregated_user_log_df2.withColumn(
    "adjusted_purchase_amount",
    when((col("date") > "2023-07-10") & (col("group") == "treatment"),
         expr("total_purchase_amount + round(rand(1) * 50+1000)"))
         .otherwise(col("total_purchase_amount"))
)
adjusted_df.display()

# COMMAND ----------

# Group by date and group, and aggregate on total_purchase_amount
grouped_df = aggregated_user_log_df2.groupBy("date", "group", "experiment_id").agg(sum("total_purchase_amount").alias("total_amount"))


# COMMAND ----------

grouped_df.display()

# COMMAND ----------

simulated_df = grouped_df.withColumn(
    "simulated_total_amount",
    when(col("date") <= "2023-07-10",
         0 + round(randn(1) * 400 + 3000))  # Add closing parenthesis here
    .when((col("date") > "2023-07-10") & (col("group") == "treatment"),
         round(randn(1) * 500 + 5000))  # Simulate increase for treatment
    .otherwise(round(randn(1) * 400 + 3000))
)

# COMMAND ----------

simulated_df.display()

# COMMAND ----------

# Calculate the daily cumulative sum by group
window_spec = Window.partitionBy("group").orderBy("date").rowsBetween(Window.unboundedPreceding, 0)
cumulative_sum_df = simulated_df.withColumn("daily_cumulative_sum", sum("simulated_total_amount").over(window_spec))
cumulative_sum_df.display()

# COMMAND ----------

def ttest_by_day(df1, df2):
    # Get a list of unique days in the dataframe
    days = df1.select('date').distinct().rdd.map(lambda r: r[0]).collect()

    # Loop through each day and perform the ztest function
    result_df = None
    for day in days:
        day_agg = df1.filter(col('date') == day)
        day_df = df2.filter(col('date') == day)
        
        # Get the control group duration array
        control_df = day_df.filter(col('group') == 'control')
        control_df = control_df.filter((col('adjusted_purchase_amount')) != 0)
        control_dura = control_df.select("adjusted_purchase_amount").agg(collect_list("adjusted_purchase_amount")).first()[0]

        # Get the treatment group duration array
        treat_df = day_df.filter(col('group') == 'treatment')
        treat_df = treat_df.filter((col('adjusted_purchase_amount')) != 0)
        treat_dura = treat_df.select("adjusted_purchase_amount").agg(collect_list("adjusted_purchase_amount")).first()[0]

        # Perform the t sample mean test
        stat, pval = ttest_ind(control_dura, treat_dura)

        # Create the result dataframe
        result_day_df = day_agg.withColumn('ab_description', 
                                          when(col('group') == 'control', 'Stats')
                                          .when(col('group') == 'treatment', 'P_value')
                                          .otherwise(None))
        result_day_df = result_day_df.withColumn('ab_value', 
                                                 when(col('ab_description') == 'Stats', stat)
                                                 .when(col('ab_description') == 'P_value', pval)
                                                 .otherwise(None))

        # Append the result dataframe to the final result dataframe
        if result_df is None:
            result_df = result_day_df
        else:
            result_df = result_df.unionByName(result_day_df)

    return result_df

# COMMAND ----------

window = Window.partitionBy("group").orderBy("date")
df = cumulative_sum_df.withColumn("experiment_duration", rank().over(window)-10)

# COMMAND ----------

df.display()

# COMMAND ----------


result = ttest_by_day(df, adjusted_df)

# COMMAND ----------

result.display()

# COMMAND ----------



# COMMAND ----------

# Save table with label as silver table
golden_tbl_name = 'averageduration_results'
result.write.format('delta').mode('overwrite').saveAsTable(f'{database_name}.{golden_tbl_name}')

# COMMAND ----------

experiment_2 = user_experiment.filter(col("experiment_id") == 'experiment_20230817_162908_575997')

# COMMAND ----------

aggregated_user_log_df2 = aggregated_user_log_df.alias('ua').join(experiment_2.alias('wa'), 
                                                     (col('ua.user_id') == col('wa.user_id')), 
                                                     how='inner')

# COMMAND ----------

aggregated_user_log_df2.display()

# COMMAND ----------

adjusted_df = aggregated_user_log_df2.withColumn(
    "adjusted_purchase_amount",
    when((col("date") > "2023-07-01") ,
         expr("total_purchase_amount + round(rand(1) * 50+500)"))
         .otherwise(col("total_purchase_amount"))
)
adjusted_df.display()

# COMMAND ----------

# Group by date and group, and aggregate on total_purchase_amount
grouped_df = aggregated_user_log_df2.groupBy("date", "group", "experiment_id").agg(sum("total_purchase_amount").alias("total_amount"))


# COMMAND ----------

grouped_df.display()

# COMMAND ----------

simulated_df = grouped_df.withColumn(
    "simulated_total_amount",
    when(col("date") <= "2023-07-03",
         0 + round(randn(1) * 400 + 3000))  # Add closing parenthesis here
    .when((col("date") > "2023-07-03") & (col("group") == "treatment"),
         round(randn(1) * 500 + 3000))  # Simulate increase for treatment
    .otherwise(round(randn(1) * 400 + 3000))
)

# COMMAND ----------

simulated_df.display()

# COMMAND ----------

# Calculate the daily cumulative sum by group
window_spec = Window.partitionBy("group").orderBy("date").rowsBetween(Window.unboundedPreceding, 0)
cumulative_sum_df = simulated_df.withColumn("daily_cumulative_sum", sum("simulated_total_amount").over(window_spec))
cumulative_sum_df.display()

# COMMAND ----------

window = Window.partitionBy("group").orderBy("date")
df = cumulative_sum_df.withColumn("experiment_duration", rank().over(window)-1)

# COMMAND ----------

result = ttest_by_day(df, adjusted_df)

# COMMAND ----------

result.display()

# COMMAND ----------

# Save table with label as silver table
golden_tbl_name = 'experiment_2_results'
result.write.format('delta').mode('overwrite').saveAsTable(f'{database_name}.{golden_tbl_name}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dashboard
# MAGIC - [AB Experiments Tracker](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/b26b684b-0858-4764-8e0d-f8e928a02811?o=1444828305810485)
# MAGIC - [AB Experiment - 1](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/ffa0d8c6-ad8e-4d38-be6b-cb16b0dc0f96?o=1444828305810485#)
# MAGIC - [AB Experiment - 2](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/e90cfe38-48ff-4e14-a737-b35ff9e74280?edit&o=1444828305810485)
# MAGIC

# COMMAND ----------


