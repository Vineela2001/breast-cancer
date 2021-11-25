# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 19:17:06 2021

@author: RISHITHA
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("FuelConsumption.csv")
#use required features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))
