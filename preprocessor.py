'''
Id:          "$Id$"
Copyright:   Copyright (c) 2018 Bank of America Merrill Lynch, All Rights Reserved
Description:
Test:
'''
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas
import requests
from pandas.stats.moments import rolling_mean
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def loadData(fileName, splitRatio=.9):
    """
    Read data, process it and split into train and test set
    """
    
    # urlInput = 'H:/test/MLChallenge/HDFC_price_vol.csv'
    urlInput = fileName
    
    datasetIn = pandas.read_csv(urlInput)
    datasetIn = preprocess(datasetIn).dropna()
    datanp = np.array(datasetIn)
    split = round(splitRatio * datasetIn.shape[0])
    
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
    
    inSet = datasetIn.icol(0)
    
    rm12 = rolling_mean(inSet,12).round(2)
    rm26 = rolling_mean(inSet,26).round(2)
    
    # MACD calculation using ma12 and ma26
    
    macd = rm12 - rm26
    macdDf = pandas.DataFrame(macd)
    df1 = pandas.concat([datasetIn,macdDf], axis=1)
    return df1
    

def main():
    X_tr, Y_tr, X_te, Y_te = loadData('H:/Sonia/MLChallenge/HDFC_price_vol.csv')
    print len(X_tr)
    print len(Y_tr)
    print len(X_te)
    print len(Y_te)

