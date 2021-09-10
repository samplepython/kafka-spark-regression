#!/usr/bin/python3                                                                                                      
                                                                                                                        
from pyspark import SparkContext                                                                                        
from pyspark.sql import SparkSession                                                                                    
from pyspark.streaming import StreamingContext                                                                          
from pyspark.streaming.kafka import KafkaUtils 
from HouseSalePricePredictor import load_csv_dataset, preprocess_data, make_mi_scores, plot_mi_scores, select_high_mi_score_features, embedding_plot,train_the_RandomForestRegressor_model,score_dataset_XGGradient
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

def handle_rdd(rdd):                                                                                                    
  if not rdd.isEmpty():                                                                                               
    global ss                                                                                                       
    df = ss.createDataFrame(rdd, schema=['text', 'predicted_price'])                                                
    df.show()                                                                                                       
    #df.write.saveAsTable(name='default.house_price_prediction', format='hive', mode='append')                                       
                                                                                                                        
sc = SparkContext(appName="HousePricePrediction")                                                                                     
ssc = StreamingContext(sc, 3)                                                                                           
                                                                                                                        
ss = SparkSession.builder.appName("HousePricePrediction").config("spark.sql.warehouse.dir", "/user/hive/warehouse").config("hive.metastore.uris", "thrift://localhost:9083").enableHiveSupport().getOrCreate()                                                                                                  
                                                                                                                        
ss.sparkContext.setLogLevel('WARN')                                                                                     
                                                                                                                        
ks = KafkaUtils.createDirectStream(ssc, ['house-price-prediction'], {'metadata.broker.list': 'localhost:9092'})  


df = load_csv_dataset('./','train.csv')

X, y = preprocess_data(df)

mi_scores = make_mi_scores(X, y)
mi_scores[::1].head(21)


plot_mi_scores(mi_scores)   

X = X[select_high_mi_score_features(mi_scores, 20)]

XGGradient_score = score_dataset_XGGradient(X,y)

# PCA Analysis
X_pca = PCA(n_components=2).fit_transform(X)
embedding_plot(X_pca, y, "PCA")

# LDA Analysis
X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
embedding_plot(X_lda, y, "LDA")

rf_model_object, X_test, y_test = train_the_RandomForestRegressor_model(X, y, 0.3)

# predicting the SalePrice on test data
predicted_sale_prices = rf_model_object.predict(X_test)

Analyse = X_test.copy()
Analyse['Actual_SalePrice'] = y_test
Analyse['Pridicted_SalePrice'] = predicted_sale_prices
Analyse['difference_in_SalePrice'] = Analyse.Actual_SalePrice - Analyse.Pridicted_SalePrice
Analyse['Accuracy-in-Percentage'] = (Analyse.Pridicted_SalePrice / Analyse.Actual_SalePrice) *  100
Analyse.to_csv('Analysis-test-data.csv')
                                                                                                                 
lines = ks.map(lambda x: x[1])                                                                       

transform = lines.map(lambda input_record: (input_record, float(rf_model_object.predict([np.fromstring(input_record[0:-1],dtype=np.float64,sep=',')])[0])))

transform.foreachRDD(handle_rdd)                                                                                        
                                                                                                                        
ssc.start()                                                                                                             
ssc.awaitTermination()

