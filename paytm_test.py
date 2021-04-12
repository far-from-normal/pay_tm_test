# %% [markdown]
# # Paytm test

# %% [markdown]
# ## Python imports
# All imports have been moved to this location to avoid clutter further down in the code.


# %% Module Imports

# pyspark imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
from pyspark.sql import functions as F
from pyspark.sql.window import Window as W
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# other imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
# import numpy as np

# set global formatting options
sns.set_style("whitegrid")
sns.set(font_scale=0.5)
mpl.rcParams['figure.dpi'] = 300
pd.options.display.float_format = '{:,.2f}'.format