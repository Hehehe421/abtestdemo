# Databricks notebook source
# MAGIC %md
# MAGIC ## What is AB testing?
# MAGIC
# MAGIC An A/B test, also called a controlled experiment or a randomized control trial, is a statistical
# MAGIC method of determining which of a set of variants is the best. A/B tests allow organizations
# MAGIC and policy-makers to make smarter, data-driven decisions that are less dependent on
# MAGIC guesswork.
# MAGIC
# MAGIC Today, A/B tests are an important business tool, used to make decisions in areas like product pricing, website design, marketing campaign design, and brand messaging. A/B testing lets organizations quickly experiment and iterate in order to continually improve their business.
# MAGIC
# MAGIC In data science, A/B tests can also be used to choose between two models in production, by measuring which model performs better in the real world. In this formulation, the control is often an existing model that is currently in production. The treatment is a new model being considered to replace the old one. We would like to using A/B test to test the effectiveness of these two model on the organization OEC (overall evaulation criteria). 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC
# MAGIC During today's demonstration, we will delve into the intricacies of constructing a machine learning (ML) model and subsequently conducting AB testing using Databricks. To provide a contextual framework for our demo, let us consider the scenario of Airbnb:
# MAGIC
# MAGIC **Demo Use Case:** Within the realm of improving the host experience on their website, Airbnb aims to enhance the process of listing creation. To achieve this, they have conceived the idea of constructing a predictive model that can accurately estimate the most suitable listing price based on various factors such as location, number of rooms, and other relevant features. This predictive model will serve as a reference point for hosts when they create new listings, ultimately streamlining the process and reducing time consumption.
# MAGIC
# MAGIC To further optimize their efforts, Airbnb has developed two distinct versions of the ML model, each employing different techniques to predict the listing price. In order to determine which version yields superior results and is more conducive to their business objectives, Airbnb intends to conduct real-time AB testing, thereby comparing the efficacy of the two models in a practical setting with actual customers.
# MAGIC
# MAGIC The primary metrics that Airbnb seeks to improve through AB testing are the website's activity rate and the total duration engaged on the website. By gauging the impact of the ML models on these metrics, Airbnb can ascertain which version better aligns with their overarching goals, ultimately fostering a more engaging and efficient platform for their users.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What expected in this Demo?

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

# MAGIC %md
# MAGIC ## Build the ML predictive models
# MAGIC In this notebook, we will focus on building and serving the model:
# MAGIC
# MAGIC 1. Model Construction and Registration: Our initial undertaking involves training four distinct regression models using the versatile ElasticNet algorithm. These models, once trained, will be registered, allowing us to efficiently track and manage their performance and associated metadata.
# MAGIC
# MAGIC 2. Pyfunc Packaging: The next crucial phase entails packaging the registered models into a cohesive pyfunc format, which empowers us to seamlessly pass parameters, particularly the user_group, thereby facilitating differentiation between the control and treatment groups. This parameter serves as a vital indicator, enabling us to meticulously analyze and compare the outcomes of the different model versions.
# MAGIC
# MAGIC 3. Online and Offline Model Serving: We will adopt a dual approach encompassing both online and offline serving. The online serving mechanism ensures real-time predictions, enabling immediate responses to user queries and demands. Conversely, the offline serving aspect focuses on batch processing, affording us the opportunity to handle larger volumes of data and perform intricate computations in a more streamlined and efficient manner.
# MAGIC

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import os
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import lit
import random
import warnings
import datetime
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
from pyspark.sql.functions import count, lit
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from scipy.stats import chisquare
from pyspark.sql.functions import rand, expr, col, when
from pyspark.sql.functions import weekofyear, date_trunc, sum, when
from scipy.stats import ttest_ind
from pyspark.sql.functions import col, sum
from scipy.stats import ttest_ind
from pyspark.sql.window import Window
from pyspark.sql.functions import sum as sql_sum
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES from abtest_workshop_sixuan_he;

# COMMAND ----------

#Import data with offline mode
bronze_tbl_name = "listing_bronze"
listings_df = spark.table(f'{database_name}.{bronze_tbl_name}')

# Drop user_id and review_score before modeling
cols_to_drop = ['user_id', 'review_score']
for col in cols_to_drop:
    if col in listings_df.columns:
        listings_df = listings_df.drop(col)
listings_df.display()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #### Featurization
# MAGIC
# MAGIC The `listings_df` DataFrame is already pretty clean, but we do have some categorical features that we'll need to convert to numeric features for modeling.
# MAGIC
# MAGIC These features include:
# MAGIC
# MAGIC * **`neighbourhood_cleansed`**
# MAGIC * **`property_type`**
# MAGIC * **`room_type`**
# MAGIC * **`instant_bookable`**
# MAGIC
# MAGIC
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Notice that we are creating a function to perform these computations. We'll use it to refer to this set of instructions when creating our feature table.

# COMMAND ----------

import pyspark.pandas as ks

def compute_features(spark_df):
  
    # Convert to Koalas DataFrame
    koalas_df = spark_df.to_koalas()

    # OHE
    ohe_koalas_df = ks.get_dummies(
      koalas_df, 
      columns=["neighbourhood_cleansed", "property_type", "room_type", "instant_bookable"],
      dtype="float64"
    )

    # Clean up column names
    ohe_koalas_df.columns = ohe_koalas_df.columns.str.replace(' ', '')
    ohe_koalas_df.columns = ohe_koalas_df.columns.str.replace('(', '-')
    ohe_koalas_df.columns = ohe_koalas_df.columns.str.replace(')', '')

    return ohe_koalas_df

# COMMAND ----------

features_df = compute_features(listings_df)
display(features_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #### Create the Feature Table
# MAGIC Our first step is to instantiate the feature store client using `FeatureStoreClient()`.
# MAGIC
# MAGIC Next, we can use the `feature_table` operation to register the DataFrame as a Feature Store table.
# MAGIC
# MAGIC In order to do this, we'll want to provide the following:
# MAGIC
# MAGIC 1. The `name` of the database and table where we want to store the feature table
# MAGIC 1. The `keys` for the table
# MAGIC 1. The `schema` of the table
# MAGIC 1. A `description` of the contents of the feature table
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This creates our feature table, but we still need to write our values in the DataFrame to the table.

# COMMAND ----------

#!pip install databricks-feature-store

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
from databricks.feature_store import feature_table
fs = FeatureStoreClient()
# drop the existing feature table
fs.drop_table(name='abtest_workshop_sixuan_he.listings_fs_sliver')

# create a new one
feature_table = fs.create_table(
  name=f"{database_name}.listings_fs_sliver",
  primary_keys=["listing_id"],
  schema=features_df.spark.schema(),
  description="This host-level table contains one-hot encoded and numeric features to predict the price of a listing."
)
fs.write_table(df=features_df.to_spark(), name=f"{database_name}.listings_fs_sliver", mode="overwrite")

# COMMAND ----------

# MAGIC %md 
# MAGIC At this point, we can head to the Feature Store UI to check out our table.
# MAGIC
# MAGIC [Feature Store - listing_fs_sliver](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#feature-store/feature-store/abtest_workshop_sixuan_he.listings_features_sliver)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build the models with MLflow

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
 
# Import mlflow
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from databricks.feature_store import FeatureLookup

# COMMAND ----------

df = spark.table(f'{database_name}.listings_fs_sliver')
df = df.toPandas()
train_x = df.drop(['price'], axis=1)
train_y = df.price

models = []
mlflow.sklearn.autolog(log_input_examples=True)
n_models = 2
for i in range(n_models):
    with mlflow.start_run() as run:
        lr = ElasticNet(alpha=0.05, l1_ratio=0.05, random_state=42)
        model = lr.fit(train_x, train_y)
        mv = mlflow.register_model(f'runs:/{run.info.run_id}/model', f'abtest_multimodel-serving-{i}')
        client = MlflowClient()
        client.transition_model_version_stage(f'abtest_multimodel-serving-{i}', mv.version, "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC [Registered Models](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#mlflow/models/abtest_multimodel-serving-3/versions/1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Packing the Multiple Model with model selection logic
# MAGIC Let's create a class called MultiModelPyfunc that extends mlflow.pyfunc.PythonModel. It serves the purpose of managing multiple models in an AB testing setting. Here's a concise summary:
# MAGIC - The `load_context` method loads the models into the instance by iterating through a range and using mlflow.sklearn.load_model to retrieve them from context artifacts.
# MAGIC - The `select_model` method determines the model to use based on the input's "group" value, returning the corresponding index.
# MAGIC - The `process_input` method preprocesses the input by dropping the "group" column, converting the DataFrame to a NumPy array, and reshaping it.
# MAGIC - The `predict` method serves as the entry point for predictions, selecting the appropriate model, processing the input, and returning the prediction.
# MAGIC
# MAGIC Overall, this code provides a concise framework for managing and utilizing multiple models in an AB testing scenario.

# COMMAND ----------

class MultiModelPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.models = []
        self.n_models = 2
        for i in range(self.n_models):
            self.models.append(mlflow.sklearn.load_model(context.artifacts[f'abtest_multimodel-serving-{i}']))
    
    def select_model(self, model_input):
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Sample model requires Dataframe inputs")
        group = model_input["group"].iloc[0]
        if group == "control":
            return 0
        elif group == "treatment":
            return 1
        else:
            raise ValueError("Group field incorrectly specified")
            
    def process_input(self, model_input):
        return model_input.drop("group", axis=1).values.reshape(1, -1)
 
    def predict(self, context, raw_input):
        selected_model = self.select_model(raw_input)
        print(f'Selected model {selected_model}')
        model = self.models[selected_model]
        model_input = self.process_input(raw_input)
        return model.predict(model_input)

# COMMAND ----------

n_models = 2
paths = []
for i in range(n_models):
    paths.append(mlflow.artifacts.download_artifacts(f'models:/abtest_multimodel-serving-{i}/Production'))
artifacts = {f'abtest_multimodel-serving-{i}': paths[i] for i in range(n_models)}

# COMMAND ----------

input_example = df.iloc[0]
input_example["group"] = "control"
input_example = input_example.to_frame().transpose()
input_example = input_example.drop("price", axis=1)
input_example

# COMMAND ----------

client = MlflowClient()
with mlflow.start_run() as run:
    model_info = mlflow.pyfunc.log_model(
      "raw-model",
      python_model=MultiModelPyfunc(),
      input_example=input_example,
      artifacts=artifacts,
    )
    model = mlflow.pyfunc.load_model(f'runs:/{run.info.run_id}/raw-model')
    prediction = model.predict(input_example)
    signature = infer_signature(input_example, prediction)
    mlflow.pyfunc.log_model(
        "augmented-model",
        python_model=MultiModelPyfunc(),
        artifacts=artifacts,
        input_example=input_example,
        signature=signature
    )
    mv = mlflow.register_model(f'runs:/{run.info.run_id}/augmented-model', "abtest_multimodel-serving")
    client.transition_model_version_stage(f'abtest_multimodel-serving', mv.version, "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC [Registered Model - abtest_multimodel-servering](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#mlflow/models/abtest_multimodel-serving)

# COMMAND ----------

model_uri = 'models:/abtest_multimodel-serving/Production'
model = mlflow.pyfunc.load_model(model_uri)

path = mlflow.artifacts.download_artifacts('models:/abtest_multimodel-serving/Production')
input_example = model.metadata.load_input_example(path)

model.predict(input_example)

# COMMAND ----------

input_example = df.iloc[0]
input_example["group"] = "treatment"
input_example = input_example.to_frame().transpose()
input_example = input_example.drop("price", axis=1)
input_example

# COMMAND ----------

model.predict(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC [Real-time serving endpoints - abtesting_multimodel_serving](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#mlflow/endpoints/abtesting_multimodel_serving)

# COMMAND ----------

# MAGIC %md
# MAGIC [AutoML](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#mlflow/experiments/3634742887752824?searchFilter=&orderByKey=metrics.%60val_r2_score%60&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All%20Runs&selectedColumns=attributes.%60Source%60,attributes.%60Models%60,tags.%60model_type%60,metrics.%60val_r2_score%60&isComparingRuns=false&compareRunCharts=)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Assign the Control and Treamtment group at user level
# MAGIC We will use a pandas UDF to split the users by a given ratio. For example, we would like to run an A/B experiment at user level for 50/50.

# COMMAND ----------

import random
from pyspark.sql import functions as F
def assign_random_group(user_bronze, ratio=0.5):
    """
    Assigns a random group to each user in the input DataFrame.

    Args:
        user_df: A PySpark DataFrame with 'host_id' and 'create_date' columns.
        ratio: A float between 0 and 1 that specifies the proportion of users to assign to the 'treatment' group.
               The remaining proportion will be assigned to the 'control' group. Default is 0.5.

    Returns:
        A new PySpark DataFrame with 'host_id', 'create_date', and 'group' columns.
    """
    groups = ['control', 'treatment']
    num_treatment = int(user_bronze.count() * ratio)
    num_control = user_bronze.count() - num_treatment

    # create the experiment variable
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    experiment = "experiment_" + timestamp

    # add the experiment column to the PySpark DataFrame
    user_bronze = user_bronze.withColumn("experiment_id", lit(experiment))

    # add the ratio column
    user_bronze = user_bronze.withColumn("experiment_ratio", lit(ratio))

    # Create a UDF to assign a random group to each row in the DataFrame
    @udf(returnType=StringType())
    def assign_group():
        return random.choice(groups)

    # Add a 'group' column to the DataFrame using the UDF
    user_bronze = user_bronze.withColumn('group', assign_group())

    # Shuffle the DataFrame and split into control and treatment groups
    user_bronze = user_bronze.orderBy('group').cache()
    control_df = user_bronze.filter(user_bronze['group'] == 'control').limit(num_control)
    treatment_df = user_bronze.filter(user_bronze['group'] == 'treatment').limit(num_treatment)


    # Combine the control and treatment groups and return the resulting DataFrame
    return control_df.union(treatment_df).select('user_id', 'group', 'experiment_id', 'experiment_ratio')

# COMMAND ----------

# assign random groups to each user in the DataFrame with a 50/50 split
bronze_tbl_name = 'user_bronze'
user_bronze = spark.table(f'{database_name}.{bronze_tbl_name}')
user_bronze = user_bronze.filter(F.col("is_host") == True)
user_experiment = assign_random_group(user_bronze, ratio=0.5)

# COMMAND ----------

user_experiment.display()

# COMMAND ----------

usersilver_tbl_name = 'user_silver'
# Create a Delta Lake table from loaded 
user_experiment.write.format('delta').mode('overwrite').option("overwriteSchema", "true").saveAsTable(f'{database_name}.{usersilver_tbl_name}')

# COMMAND ----------

# assign random groups to each user in the DataFrame with a 50/50 split
bronze_tbl_name = 'user_bronze'
user_bronze = spark.table(f'{database_name}.{bronze_tbl_name}')
#user_bronze = user_bronze.filter(F.col("is_host") == True)
user_experiment2 = assign_random_group(user_bronze, ratio=0.2)

# COMMAND ----------

user_experiment2.display()

# COMMAND ----------

usersilver_tbl_name = 'user_silver'
# Create a Delta Lake table from loaded 
user_experiment2.write.format('delta').mode('append').option("overwriteSchema", "true").saveAsTable(f'{database_name}.{usersilver_tbl_name}')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Some Practical Considerations
# MAGIC
# MAGIC **Splitting your subjects**: When splitting your subjects up randomly between models, make sure the process is truly random, and think through any interference between the two groups. Do they communicate or influence each other in some way? Does the randomization method cause an unintended bias? Any bias in group assignments can invalidate the results. Also, make sure the assignment is consistent so that each subject always gets the same treatment. For example, a specific customer should not get different prices every time they reload the pricing page. 
# MAGIC
# MAGIC **A/A Tests**: It can be a good idea to run an A/A test, where both groups are control or treatment groups. This can help surface unintentional biases or errors in the processing and can give a better feeling for how random variations can affect intermediate results. 
# MAGIC
# MAGIC **Don’t Peek!**: Due to human nature, it’s difficult not to peek at the results early and draw conclusions or stop the experiment before the minimum sample size is reached. Resist the temptation. Sometimes the “wrong” model can get lucky for a while. You want to run a test long enough to be confident that the behavior you see is really representative and not just a weird fluke. 
# MAGIC
# MAGIC The more sensitive a test is, the longer it will take: The resolution of an A/B test (how small a delta effect size you can detect) increases as the square of the samples. In other words, if you want to halve the delta effect size you can detect, you have to quadruple your sample size.
