#!/usr/bin/python3  

import pandas as pd
from kafka import KafkaProducer                                                                                         
from random import randint                                                                                              
from time import sleep
import numpy as np                                                                                                  
import sys  

BROKER = 'localhost:9092'                                                                                               
TOPIC = 'house-price-prediction' 

try:                                                                                                                    
    p = KafkaProducer(bootstrap_servers=BROKER)                                                                         
except Exception as e:                                                                                                  
    print(f"ERROR --> {e}")                                                                                             
    sys.exit(1) 

X = pd.read_csv('test.csv')

#-----Feature Engineering-----
X['TotalSquareFootage'] = X['TotalBsmtSF'] + X['GrLivArea']

# Converting float64 and categorical to int64
# float_columns = X.select_dtypes(np.float64)
# LotFrontage MasVnrArea GarageYrBlt
for float_column in X.select_dtypes(np.float64):
   X[float_column] = X[float_column].fillna(0).astype(np.int64)
   X[float_column].astype(np.int64)

 # Label encoding for categoricals
for colname in X.select_dtypes("object"):
   X[colname], _ = X[colname].factorize()

#print(X.head(20))

lst_cols = ['TotalSquareFootage','OverallQual','GrLivArea','GarageCars','BsmtQual','KitchenQual','ExterQual','GarageArea','YearBuilt','GarageFinish','TotalBsmtSF','FullBath','1stFlrSF','GarageType','YearRemodAdd','Foundation','TotRmsAbvGrd','Fireplaces','HeatingQC','Neighborhood']
 
X = X[lst_cols]


while True:
  for row_index in range(len(X)):
    str_row=''
    for col_index in range(len(X.columns)):
     str_row += str(X.iloc[row_index,col_index])
     str_row += ','  
    print(str_row)  
    p.send(TOPIC, bytes(str_row, encoding="utf8"))     
    sleep(randint(1,4))
    print('----------------------')        

 
  
