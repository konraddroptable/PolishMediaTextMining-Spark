
def sortArr(x):
    x.sort(key = lambda y: y[0])
    
    return x

def wordCountPerSourceArticle(df, articleSource):
    wc = df.where(df.source == articleSource) \
        .select(col("text"), col("source")) \
        .rdd \
        .zipWithIndex() \
        .map(lambda row: Row(rowId = row[1], source = row[0][1], text = row[0][0])) \
        .toDF() \
        .flatMap(lambda row: [str(row.rowId) + ' ' + \
                                row.source.replace('\"', '') + ' ' + \
                                word for word in row.text.replace('\"', '').split()]) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .toDF() \
        .map(lambda line: Row(rowId = int(line._1.split()[0]), \
                            source = line._1.split()[1], \
                            word = line._1.split()[2], \
                            count = line._2)) \
        .toDF() \
        .cache()
        
    return wc

def uniqueWords(wc, articleSource):
    uwords = wc.where(wc.source == articleSource) \
            .select(col("word").alias("word_u"), col("count")) \
            .groupBy("word_u") \
            .agg(sum("count").alias("count")) \
            .rdd \
            .zipWithIndex() \
            .map(lambda row: Row(wordId = row[1], word_u = row[0][0], count_all = row[0][1])) \
            .toDF() \
            .cache()
            
    return uwords

def createCorpus(wc, uwords, articleSource):
    nwords = uwords.select("wordId").distinct().count()
    lda_entries = wc.where(wc.source == articleSource) \
                    .join(uwords, wc.word == uwords.word_u) \
                    .select(col("rowId").alias("doc_id"), col("wordId").alias("word_id"), col("count")) \
                    .rdd \
                    .map(lambda row: (row[0], [row[1], float(row[2])])) \
                    .groupByKey() \
                    .mapValues(lambda x: list(x)) \
                    .cache()
    sortEntries = lda_entries.map(lambda line: (line[0], sortArr(line[1]))).cache()
    corpus = sortEntries.map(lambda row: [row[0], SparseVector(nwords, \
                                                    map(lambda x: x[0], row[1]), \
                                                    map(lambda x: x[1], row[1]))])
    
    return corpus

def fitModel(corpus, k, maxIterations=10):
    return LDA.train(corpus, k=k, maxIterations=maxIterations)

def describeTopicsFromModel(model, uwords, maxTermsPerTopic=15):
    topics = model.describeTopics(maxTermsPerTopic)
    res = sc.parallelize(topics) \
        .zipWithIndex() \
        .map(lambda row: [Row(topicId = row[1], wordId = row[0][0][i], weight = row[0][1][i]) for i in range(maxTermsPerTopic)]) \
        .reduce(lambda a, b: a + b)
    summ_frm = sc.parallelize(res).toDF()
    summary = summ_frm.join(uwords, summ_frm.wordId == uwords.wordId).drop("wordId").toPandas()
    
    return summary

def summaryOnAllSources(df, sources):
    wc = wordCountBySourcePerArticle(df)
    return reduce(lambda a, b: pd.concat([a, b], axis=0), map(lambda source: summaryPerSource(wc, source), sources))

def summaryPerSource(df, articleSource):
    print "Wordcount " + str(articleSource)
    wc = wordCountPerSourceArticle(df, articleSource)
    print "Corpus creation " + str(articleSource)
    uwords = uniqueWords(wc, articleSource)
    corpus = createCorpus(wc, uwords, articleSource)
    print "Model training " + str(articleSource)
    model = fitModel(corpus, 8, 100)
    print "Printing results " + str(articleSource)
    summary = describeTopicsFromModel(model, uwords)
    summary["source"] = articleSource
    
    return summary

