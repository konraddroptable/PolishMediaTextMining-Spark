#### Q = Nasz_Dziennik, P = Gazeta_Wyborcza
def getDKLOnSource(Pprob, Qprob):
    PQ = Pprob.join(Qprob, Pprob.word_p == Qprob.word_q)
    DKL = PQ.withColumn("PQ", Pprob.prob_P * log(Pprob.prob_P / Qprob.prob_q)) \
            .select("PQ") \
            .fillna(0) \
            .agg(sum("PQ").alias("DKL")) \
            .toPandas() # DKL(P=GW||Q=ND) = 180
    
    return DKL

def getQprobOnSource(df, articleSource):
    Q = df.where(df.source == articleSource) \
        .flatMap(lambda row: row.text.replace('\"', '').split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .toDF() \
        .select(col("_1").alias("word"), col("_2").alias("count")) \
        .fillna(0) \
        .cache()

    Qcount = Q.select(col("count")).map(lambda row: row[0]).reduce(lambda a, b: a + b)
    Qprob = Q.rdd.map(lambda row: Row(**dict(row.asDict(), prob = row["count"] / Qcount))) \
                .toDF() \
                .select(col("word").alias("word_q"), col("prob").alias("prob_q")) \
                .cache()
    
    return Qprob

def getPprobOnSource(df, articleSource):
    P = df.where(df.source == articleSource) \
        .flatMap(lambda row: [row.source.replace('\"', '') + ' ' + word.replace('\"', '') for word in row.text.split()]) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .toDF() \
        .map(lambda line: (line._1.split(" ")[0], line._1.split(" ")[1], line._2)) \
        .toDF() \
        .select(col("_1").alias("source"), col("_2").alias("word"), col("_3").alias("count")) \
        .groupBy("word") \
        .pivot("source") \
        .sum("count") \
        .fillna(0) \
        .cache()

    Pcount = P.count()
    
    Pprob = P.rdd.map(lambda row: Row(**dict(row.asDict(), \
                                        prob_P = row[articleSource.replace('\"', '')] / Pcount))) \
                .toDF() \
                .select(col("word").alias("word_p"), col("prob_P")) \
                .cache()
        
    return Pprob



#### On all articles
def getQprob(df):
    Q = df.flatMap(lambda row: row.text.replace('\"', '').split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .toDF() \
        .select(col("_1").alias("word"), col("_2").alias("count")) \
        .fillna(0) \
        .cache()

    Qcount = Q.select(col("count")).map(lambda row: row[0]).reduce(lambda a, b: a + b)
    Qprob = Q.rdd.map(lambda row: Row(**dict(row.asDict(), prob = row["count"] / Qcount))) \
                .toDF() \
                .select(col("word").alias("word_q"), col("prob").alias("prob_q")) \
                .cache()
    
    return Qprob


def getPprobOnAll(df):
    P = df.flatMap(lambda row: [row.source.replace('\"', '') + ' ' + word.replace('\"', '') for word in row.text.split()]) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .toDF() \
        .map(lambda line: (line._1.split(" ")[0], line._1.split(" ")[1], line._2)) \
        .toDF() \
        .select(col("_1").alias("source"), col("_2").alias("word"), col("_3").alias("count")) \
        .groupBy("word") \
        .pivot("source") \
        .sum("count") \
        .fillna(0) \
        .cache()
        
    Pcount = P.count()
    Pprob = P.rdd.map(lambda row: Row(**dict(row.asDict(), \
                                        prob_GW = row["Gazeta_Wyborcza"] / Pcount, \
                                        prob_ND = row["Nasz_Dziennik"] / Pcount, \
                                        prob_SE = row["Super_Express"] / Pcount))) \
                .toDF() \
                .select(col("word").alias("word_p"), col("prob_GW"), col("prob_ND"), col("prob_SE")) \
                .cache()
    
    return Pprob
    
 
def getDKLOnAll(Pprob, Qprob):
    PQ = Pprob.join(Qprob, Pprob.word_p == Qprob.word_q)
    PQ = PQ.withColumn("GW", PQ.prob_GW * log(PQ.prob_GW / PQ.prob_q)) \
            .withColumn("ND", PQ.prob_ND * log(PQ.prob_ND / PQ.prob_q)) \
            .withColumn("SE", PQ.prob_SE * log(PQ.prob_SE / PQ.prob_q)) \
            .select("GW", "ND", "SE") \
            .cache()

    DKL = PQ.agg(sum("GW").alias("DKL_GW"), sum("ND").alias("DKL_ND"), sum("SE").alias("DKL_SE")).toPandas()
    
    return DKL   

    
    
    
    