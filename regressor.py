'''
Id:          "$Id$"
Copyright:   Copyright (c) 2018 Bank of America Merrill Lynch, All Rights Reserved
Description:
Test:
'''

from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from playground.sonia.preprocessor import loadData


class Regressor(object):
    
    def __init__(self, X_tr, Y_tr, X_te, Y_te, randomSeed=None, hiddenLayers=10, neuronsPerLayer=100):
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.X_te = X_te
        self.Y_te = Y_te
        self.randomSeed = np.random.seed(10)
        # self.randomSeed = None
        self.hiddenLayers = hiddenLayers
        self.neuronsPerLayer = neuronsPerLayer
        
        self.reg = MLPRegressor(hidden_layer_sizes=(10,20),  activation='relu', solver='lbfgs', alpha=0.001,batch_size='auto',
                   learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100, shuffle=True,
                   random_state=self.randomSeed, tol=0.0001, verbose=False, warm_start=True, momentum=0.9,
                   nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=.9, beta_2=0.8,
                   epsilon=1e-08)
               
    def fit(self, X_tr, Y_tr):
        self.reg.fit(X_tr, Y_tr)
    
    def predict(self, X_te):
        return self.reg.predict(X_te)
            
    def plot(self,Y_te,Y_predicted):
        t = np.arange(0, 24, 1)
        p1, = plt.plot(t, Y_te, 'bs')
        p2, = plt.plot(t, Y_predicted, 'r^')
        plt.legend([p1, p2],['Actual Price','Predicted Price'])
        plt.show()
        
    def score(self, X_te,Y_te):
        return self.reg.score(X_te,Y_te)
    
def main():
    X_tr, Y_tr, X_te, Y_te = loadData('H:/Sonia/MLChallenge/HDFC_price_vol.csv')
    reg = Regressor(X_tr, Y_tr, X_te, Y_te)
    reg.fit(X_tr, Y_tr)
    Y_predicted = reg.predict(X_te)
    #reg.plot(Y_te,Y_predicted)
    score = reg.score(X_te, Y_te)
    print score
    
