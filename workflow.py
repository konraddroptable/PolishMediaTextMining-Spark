from __future__ import division
import pandas as pd
import numpy as np
from numpy.linalg import eigh

from pyspark import SparkContext, SQLContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import SparseVector, DenseVector, Vector, Vectors



######### Launch spark local session on 4 cores
# more info:
# http://spark.apache.org/docs/latest/configuration.html
conf = (SparkConf().set("spark.executor.memory", "2g") \
                    .set("spark.driver.memory", "2g") \
                    .set("spark.driver.maxResultSize", "2g"))
sc = SparkContext(master="local[4]", conf=conf)
sqlContext = SQLContext(sc)

######### Paths
class ProjectPaths:
    project_directory = '/home/konrad/Desktop/Python scripts/'
    data_folder = '/home/konrad/Desktop/Python scripts/data/'


######### Import data
# Lemmatized articles
rdd = sc.textFile(ProjectPaths.data_folder + "clean_articles.csv", minPartitions=4)
rdd = rdd.map(lambda line: line.split(";"))
header = rdd.first()
rdd = rdd.filter(lambda line: line != header)
df = rdd.map(lambda line: Row(date = line[0].replace('\"', ''), \
                            title = line[1], \
                            text = line[2], \
                            source = line[3].replace('\"', ''))) \
        .toDF()


######### LDA model
"""
Params:
k=8
maxIter=100
maxTermsPerTopic=15
"""
# run script "lda.py"
gw = summaryPerSource(df, "Gazeta_Wyborcza")
gw.to_csv(ProjectPaths.data_folder + "lda_gw.csv", index = False)

nd = summaryPerSource(df, "Nasz_Dziennik")
nd.to_csv(ProjectPaths.data_folder + "lda_nd.csv", index = False)

se = summaryPerSource(df, "Super_Express")
se.to_csv(ProjectPaths.data_folder + "lda_se.csv", index = False)




######### PCA
"""
Params:
k=2
"""
# run script "pca.py"

events = pd.read_table(ProjectPaths.data_folder + "daty.txt", sep=";")
frm_events = createEventsFrame(events)
keywords = sqlContext.createDataFrame(map(lambda row: (row[0], row[1]), frm_events.values), ["date_event", "key"])

pcaItems = fitPCAmodel(df)

pcaItems.pca_results.to_csv(ProjectPaths.data_folder + "pca_output.csv", index = False)
pcaItems.events_wordcount.to_csv(ProjectPaths.data_folder + "events_wordcount.csv", index=False)


######### Kullback-Leibler divergence
#run script "kl divergence.py"

Qprob = getQprob(df)
Pprob = getPprobOnAll(df)

DKL_All = getDKLOnAll(Pprob, Qprob)


Qprob_ND = getQprobOnSource(df, 'Nasz_Dziennik')
Pprob_GW = getPprobOnSource(df, 'Gazeta_Wyborcza')

DKL_Source = getDKLOnSource(Pprob_GW, Qprob_ND)


