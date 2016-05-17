class PCAitems(object):
    def __init__(self, pca_results, events_wordcount):
        self.pca_results = pca_results
        self.events_wordcount = events_wordcount

# Principal Component Analysis algorithm implementation
class PCA(object):
    def __init__(self, df, k):
        self.df = df
        self.k = k

    def estimateCovariance(self):
        m = self.df.select(self.df['features']).map(lambda x: x[0]).mean()
        dfZeroMean = self.df.select(self.df['features']).map(lambda x: x[0]).map(lambda x: x-m)
        self.cov = dfZeroMean.map(lambda x: np.outer(x, x)).sum() / self.df.count()
    
    def fit(self):
        col_len = self.cov.shape[1]
        eigVals, eigVecs = eigh(self.cov)
        self.eigenValues_Unsorted = eigVals
        inds = np.argsort(eigVals)
        self.eigenValues_Order = inds
        eigVecs = eigVecs.T[inds[-1:-(col_len+1):-1]]
        components = eigVecs[0:self.k]
        eigVals = eigVals[inds[-1:-(col_len+1):-1]]  # sort eigenvals
        self.eigenValues_Sorted = eigVals
        score = self.df.select(self.df['features']).map(lambda x: x[0]).map(lambda x: np.dot(x, components.T))
        # Return the `k` principal components, `k` scores, and all eigenvalues
        
        return components.T, score, eigVals
    
    def varianceExplained(self, sort=True):
        if sort:
            return self.eigenValues_Sorted / np.sum(self.eigenValues_Sorted)
        else:
            return self.eigenValues_Unsorted / np.sum(self.eigenValues_Unsorted)


def fitPCAmodel(df):
    print "Data transformations"
    wordcount_event = wordcountPerEvent(df, keywords)
    pca_event = eventsFrame(wordcount_event)
    #events_wordcount = wordcountPerEventFrame(pca_event) #very slow performance
    pca_input = pcaDataInput(pca_event)
    
    pca = PCA(pca_input, k=2)
    print "Covariance matrix estimation"
    pca.estimateCovariance()
    comp, score, eigVals = pca.fit()
    
    print "Printing results"
    pca_results = pd.DataFrame({"x": map(lambda row: row[0], comp), \
                                "y": map(lambda row: row[1], comp), \
                                "names": pca_event.columns[1:]})
    
    pcaItems = PCAitems(pca_results, events_wordcount)
    
    return pcaItems

def pcaDataInput(pca_event):
    pca_tab = pca_event.drop(col("word")).map(lambda row: (Vectors.dense([x for x in row]), ))
    pca_frm = sqlContext.createDataFrame(pca_tab, ["features"])
    
    return pca_frm
    
def wordcountPerEventFrame(pca_event):
    events_wordcount = reduce(lambda frame1, frame2: pd.concat([frame1, frame2]), \
                    map(lambda colName: createWordcountFrame(colName, 10), pca_event.columns[1:]))
    
    return events_wordcount


def eventsFrame(wordcount_event):
    pca_event = wordcount_event.withColumn("key", concat(col("source"), lit("__"), col("event"))) \
                .select(col("key"), col("word"), col("count")) \
                .groupBy(col("word")) \
                .pivot("key") \
                .sum("count") \
                .fillna(0) \
                .cache()
    
    return pca_event

def wordcountPerEvent(df, keywords):
    rdd_joined = df.join(keywords, df.date == keywords.date_event)
    
    wordcount_event = rdd_joined.flatMap(lambda row: \
                                [row.source.replace('\"', '') + ' ' \
                                    + row.key + ' ' \
                                    + word.replace('\"', '') for word in row.text.split(" ")]) \
                            .map(lambda word: (word, 1)) \
                            .reduceByKey(lambda a, b: a + b) \
                            .toDF() \
                            .cache() \
                            .map(lambda line: (line._1.split(" ")[0], line._1.split(" ")[1], line._1.split(" ")[2], line._2)) \
                            .toDF() \
                            .select(col("_1").alias("source"), \
                                    col("_2").alias("event"), \
                                    col("_3").alias("word"), \
                                    col("_4").alias("count")) \
                            .cache()
    
    return wordcount_event

def createDateTimeSeries(date, key):
    st = pd.bdate_range(end=date, periods=8, freq="D").map(lambda x: x.strftime("%d.%m.%Y"))
    en = pd.bdate_range(start=date, periods=8, freq="D").map(lambda x: x.strftime("%d.%m.%Y"))
    
    st_df = pd.DataFrame(map(lambda x: str(x)[0:10], st[0:len(st)-1]), columns=["date_event"])
    en_df = pd.DataFrame(map(lambda x: str(x)[0:10], en[1:len(en)]), columns=["date_event"]) 
    st_df["key"] = key + "_start"
    en_df["key"] = key + "_end"
    
    return pd.concat([st_df, en_df], axis=0)

def createEventsFrame(events):
    df_events = reduce(lambda row1, row2: pd.concat([row1, row2]), \
        map(lambda row_id: createDateTimeSeries(events.ix[row_id]["data"], events.ix[row_id]["id"]), \
            range(events.shape[0])))
    
    return df_events

def createWordcountFrame(columnName, size):
    data = pca_event \
            .select(col("word"), col(columnName)) \
            .sort(desc(columnName)) \
            .take(size)
    fields = data[0].__fields__
    idx = range(len(data) + 1)[1:] #1-based index
    key_split = fields[1].split("__")
    ret = pd.DataFrame(map(lambda row: [key_split[0], key_split[1], row[fields[0]], row[fields[1]]], data), \
                        columns = ["source", "event", "word", "count"])
    ret["position"] = idx
    
    return ret