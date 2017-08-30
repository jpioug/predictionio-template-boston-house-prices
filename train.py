# coding: utf-8

### BEGIN: SETUP ###
import atexit
import platform

import py4j
import sys

import pyspark
from pyspark.context import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.storagelevel import StorageLevel
from pypio.utils import new_string_array
from pypio.data import PEventStore


SparkContext._ensure_initialized()
try:
    SparkContext._jvm.org.apache.hadoop.hive.conf.HiveConf()
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
except py4j.protocol.Py4JError:
    spark = SparkSession.builder.getOrCreate()
except TypeError:
    spark = SparkSession.builder.getOrCreate()

sc = spark.sparkContext
sql = spark.sql
atexit.register(lambda: sc.stop())

sqlContext = spark._wrapped
sqlCtx = sqlContext

p_event_store = PEventStore(spark._jsparkSession, sqlContext)


def run_pio_workflow(model):
    template_engine = sc._jvm.org.jpioug.template.python.Engine
    template_engine.modelRef().set(model._to_java())
    main_args = new_string_array(sys.argv, sc._gateway)
    create_workflow = sc._jvm.org.apache.predictionio.workflow.CreateWorkflow
    sc.stop()
    create_workflow.main(main_args)

### END: SETUP ###

# In[ ]:


from pyspark.sql.functions import col
from pyspark.sql.functions import explode
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import IndexToString
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ## Load Data from PIO

# In[ ]:


event_df = p_event_store.find('BHPApp')


# In[ ]:


event_df.show(5)


# In[ ]:


def get_field_type(name):
    return 'double'

field_names = (event_df
            .select(explode("fields"))
            .select("key")
            .distinct()
            .rdd.flatMap(lambda x: x)
            .collect())
field_names.sort()
exprs = [col("fields").getItem(k).cast(get_field_type(k)).alias(k) for k in field_names]
data_df = event_df.select(*exprs)
data_df = data_df.withColumnRenamed("MEDV", "label")


# In[ ]:


data_df.show(5)


# ## Train and Test

# In[ ]:


(train_df, test_df) = data_df.randomSplit([0.9, 0.1])


# In[ ]:


featureAssembler = VectorAssembler(inputCols=[x for x in field_names if x != 'MEDV'],
                                   outputCol="rawFeatures")
scaler = StandardScaler(inputCol="rawFeatures", outputCol="features")
# TODO NPE
# clf = DecisionTreeRegressor(featuresCol="features", labelCol="label", predictionCol="prediction",
#                             maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
#                             maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10,
#                             impurity="variance", seed=None, varianceCol=None)
# clf = DecisionTreeRegressor()
clf = RandomForestRegressor(featuresCol="features", labelCol="label", predictionCol="prediction",
                            maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
                            maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10,
                            impurity="variance", subsamplingRate=1.0, seed=None, numTrees=20,
                            featureSubsetStrategy="auto")
# TODO NPE
# clf = LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction",
#                        maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True,
#                        standardization=True, solver="auto", weightCol=None, aggregationDepth=2)
# clf = LinearRegression()
# clf = GBTRegressor(featuresCol="features", labelCol="label", predictionCol="prediction",
#                    maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,
#                    cacheNodeIds=False, subsamplingRate=1.0, checkpointInterval=10, lossType="squared",
#                    maxIter=20, stepSize=0.1, seed=None
# TODO NPE
# clf = GeneralizedLinearRegression(labelCol="label", featuresCol="features", predictionCol="prediction",
#                                   family="gaussian", link=None, fitIntercept=True, maxIter=25, tol=1e-6,
#                                   regParam=0.0, weightCol=None, solver="irls", linkPredictionCol=None)
# clf = GeneralizedLinearRegression()
pipeline = Pipeline(stages=[featureAssembler, scaler, clf])


# In[ ]:


model = pipeline.fit(train_df)


# In[ ]:


predict_df = model.transform(test_df)


# In[ ]:


predict_df.select("prediction", "label").show(5)


# In[ ]:


evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predict_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[ ]:

run_pio_workflow(model)
