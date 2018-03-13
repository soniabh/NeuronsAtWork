
# coding: utf-8

# In[45]:


from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas
import requests
from pandas.stats.moments import rolling_mean
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def loadData(fileName, splitRatio=.7, beta=0):
    """
    Read data, process it and split into train and test set
    """
    
    # urlInput = 'H:/test/MLChallenge/HDFC_price_vol.csv'
    urlInput = fileName
    
    datasetIn = pandas.read_csv(urlInput)
    datasetIn = preprocess(datasetIn).dropna()
    data_obv = calculateOBV(datasetIn)
    obv = pandas.Series(data_obv, name="OBV")
    df1 = pandas.concat([datasetIn,obv], axis=1)
    df1["betas"] = beta
    #betas = pandas.Series(betas,name = "beta")
    #df1 = pandas.concat([df1,betas], axis=1)
    #df1 = df1.add(3,"beta")
    datanp = np.array(df1.dropna())
    split = round(splitRatio * df1.shape[0])
    print df1
    
    train = datanp[:int(split), :]
    test = datanp[int(split)+1:,:]
    
    X_tr = train[:-1, :]
    Y_tr = train[1:, 0]
    X_te = test[:-1, :]
    Y_te = test[1:, 0]
    
    return X_tr, Y_tr, X_te, Y_te
    

def preprocess(datasetIn):
    """
    calculates 12 day and 26 day moving averages and MACD using them.
    """
    
    inSet = datasetIn["Adj Close"]
    
    rm12 = rolling_mean(inSet,12).round(2)
    rm26 = rolling_mean(inSet,26).round(2)
    
    # MACD calculation using ma12 and ma26
    
    macd = rm12 - rm26
    macdDf = pandas.DataFrame(macd)
    mac = pandas.Series(macd,name="MACD")
    df1 = pandas.concat([datasetIn,mac], axis=1)
    return df1
    
    
def calculateOBV(datasetIn):
     
    d = datasetIn
    close_data = np.array(d["Adj Close"])
    #print close_data
    volume = np.array(d["Volume"])
    obv = preprocessing.scale(volume)
    print len(volume)

    obv = np.zeros(len(volume))
    obv[0] = 1
    print len(obv)

    for idx in range(1, len(obv)):
        if close_data[idx] > close_data[idx-1]:
            obv[idx] = obv[idx-1] + volume[idx]
        elif close_data[idx] < close_data[idx-1]:
            obv[idx] = obv[idx-1] - volume[idx]
        elif close_data[idx] == close_data[idx-1]:
                obv[idx] = obv[idx-1]
    
    return obv
    #TODO: might need to scale OBVs since they are large numbers            
    #scaled_obv = preprocessing.scale(obv)
    #print obv

print "Sonia"
X_tr, Y_tr, X_te, Y_te = loadData('C:\Users\hcl\Downloads\hdfc.csv')
print X_tr
    

    
'''
print len(X_tr)
print len(Y_tr)
print len(X_te)
print len(Y_te)
'''


