import pandas as pd
import numpy as np
concrete_data =pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data_colums= concrete_data.columns
# print(concrete_data.columns)
predictors=concrete_data[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',
       'Coarse Aggregate', 'Fine Aggregate', 'Age']]
target = concrete_data['Strength']
# print (predictors.head())
predictors_norm = (predictors - predictors.mean()) / predictors.std()
n_cols = predictors_norm.shape[1]
import keras
from keras.models import Sequential
from keras.layers import Dense
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
model = regression_model()
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)