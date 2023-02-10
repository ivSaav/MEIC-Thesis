import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner import RandomSearch, Hyperband, BayesianOptimization
from keras import models, layers
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from numpy.random import seed
from keras.utils.vis_utils import plot_model
from pickle import load
from sklearn.model_selection import train_test_split

# Define parameters of HyperModel 
class RegressionHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape    

    def build(self, hp):
        model = Sequential()
        model.add(layers.Flatten(input_shape=self.input_shape))

        for i in range(hp.Int('layers',5,100)):
          model.add(layers.Dense(units=hp.Int('units_' + str(i), 16, 128, step=4), activation=hp.Choice('act_' + str(i), ['relu', 'tanh', 'sigmoid'])))
                  #########################
          model.add(
            layers.Dropout(
              hp.Float('dropout', min_value=0.1, max_value=0.2, default=0.1, step=0.01)
            )
          )  
        model.add(layers.Dense(1920))

        model.compile(
            # optimizer=optimizers.Adam( hp.Choice("learning_rate", values=[1e-2, 1e-3])),
            optimizer='adam', loss='mse', metrics=['mse']
        )
    
        return model




# #load data and scalers
# data_inputs = pd.read_csv("./MULTI_VP_profiles_Compiled/inputsdata_compilation.csv")
# data_outputs = pd.read_csv("./MULTI_VP_profiles_Compiled/outputs_compiled.csv")
# scaler_inputs = load(open('./Models/scaler_inputs.pkl', 'rb'))
# scaler_outputs = load(open('./Models/scaler_outputs.pkl', 'rb'))
# print("loaded data and scalers from disk")


# #splitting and normalizing the data

# trainX, testX, trainY, testY = train_test_split(data_inputs, data_outputs, test_size=0.15, random_state=1)
# trainX, valX, trainY, valY = train_test_split(data_inputs, data_outputs, test_size=0.15, random_state=1)


# testX = scaler_inputs.transform(testX)
# trainX = scaler_inputs.transform(trainX)
# testY = scaler_outputs.transform(testY)
# trainY = scaler_outputs.transform(trainY)
# #print("Inputs: Train \n",testX, trainX.shape, "\n Test \n", testX, testX.shape)
# #print("Outputs: Train \n", trainY, trainY.shape, "\n Test\n", testY, testY.shape)
# print("split and normalized data")






# #--------------------Random Search--------------------
# #hypermodel training
# input_shape = (trainX.shape[1],)
# hypermodel = RegressionHyperModel(input_shape)

# tuner_rs = RandomSearch(
#             hypermodel,
#             objective='mse',
#             seed=42,
#             max_trials=100,
#             executions_per_trial=1,
#             overwrite=True
#             )

# tuner_rs.search(trainX, trainY, epochs=500, validation_split=0.2, verbose=1)  #epochs=500

# best_model_r = tuner_rs.get_best_models(num_models=1)[0]
# loss_r, mse_r = best_model_r.evaluate(testX, testY)
# print("Random Search")
# print(best_model_r.summary())
# print(loss_r,mse_r)
# best_model_r.save("./Models/Tuner_Models/random_search_model")

