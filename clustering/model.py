from keras_tuner import HyperModel
from keras import layers
from keras.models import Sequential, load_model
import pandas as pd
from pickle import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from keras_tuner import RandomSearch
from keras.callbacks import EarlyStopping

from tools.data import plot_data_values
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.utils import shuffle

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# Define parameters of HyperModel 
class RegressionHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape    

    def build(self, hp):
        model = Sequential()
        model.add(layers.Flatten(input_shape=self.input_shape))

        for i in range(hp.Int('layers',8,40)):
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
            optimizer='adam',loss='mse',metrics=['mse']
        )
    
        return model
      
      
if __name__ == "__main__":
    #load data and scalers
    # data_inputs = pd.read_csv("../data/compiled/inputsdata_compilation.csv")
    # data_outputs = pd.read_csv("../data/compiled/outputs_compiled.csv")
    # scaler_inputs = load(open('./scalers/scaler_inputs.pkl', 'rb'))
    # scaler_outputs = load(open('./scalers/scaler_outputs.pkl', 'rb'))
    # print("loaded data and scalers from disk")
    
    data_inputs = pd.read_csv("../data/compiled/inputs.csv")
    data_outputs = pd.read_csv("../data/compiled/outputs_inter.csv")
    
    data_inputs, data_outputs = shuffle(data_inputs, data_outputs, random_state=1)
    
    # in_filenames = data_inputs[['filename']]
    # out_filenames = data_outputs[['filename']]
    
    val_files = []
    with open("../data/testing_profiles.txt", "r") as f:
      val_files = f.readlines()
      val_files = [f.split(".")[0] for f in val_files]
    
    valX = data_inputs[data_inputs['filename'].isin(val_files)]
    valY = data_outputs[data_outputs['filename'].isin(val_files)]
    data_inputs = data_inputs[~data_inputs['filename'].isin(val_files)]
    data_outputs = data_outputs[~data_outputs['filename'].isin(val_files)]
    
    data_inputs = data_inputs.drop(['filename'], axis=1)
    data_outputs = data_outputs.drop(['filename'], axis=1)
    valX.drop(['filename'], axis=1, inplace=True)
    valY.drop(['filename'], axis=1, inplace=True)
    
    scaler_inputs = QuantileTransformer()
    data_inputs = scaler_inputs.fit_transform(data_inputs)
    valX = scaler_inputs.transform(valX)
    scaler_outputs = QuantileTransformer()
    data_outputs = scaler_outputs.fit_transform(data_outputs)
    valY = scaler_outputs.transform(valY)

    trainX, testX, trainY, testY = train_test_split(data_inputs, data_outputs, test_size=0.15, random_state=1, shuffle=True)
    # _trainX, valX, _trainY, valY = train_test_split(data_inputs, data_outputs, test_size=0.40, random_state=1, shuffle=True)

    #print("Inputs: Train \n",testX, trainX.shape, "\n Test \n", testX, testX.shape)
    #print("Outputs: Train \n", trainY, trainY.shape, "\n Test\n", testY, testY.shape)
    print("split and normalized data")


    #--------------------Random Search--------------------
    #hypermodel training
    input_shape = (trainX.shape[1],)
    hypermodel = RegressionHyperModel(input_shape)

    tuner_rs = RandomSearch(
                hypermodel,
                objective='mse',
                seed=42,
                max_trials=40,
                executions_per_trial=1,
                overwrite=True,
                )

    tuner_rs.search(trainX, trainY, epochs=500, validation_split=0.2, verbose=1, callbacks=[early_stop])  #epochs=500

    best_model_r = tuner_rs.get_best_models(num_models=1)[0]
    loss_r, mse_r = best_model_r.evaluate(testX, testY)
    print("Random Search")
    print(best_model_r.summary())
    print(loss_r,mse_r)
    # best_model_r.save("./out/haha/random_search.h5")
    
    
    # model = best_model_r
    
    
    files = [f for f in Path("../data/processed").iterdir() ]
    
    
    for idx, model in enumerate(tuner_rs.get_best_models(num_models=25)):
      cnt = 0
      
      #Normalize the inputs
      #predictions
      bla = model.predict(data_inputs)
      preds = pd.DataFrame(bla)
      preds.columns = scaler_outputs.feature_names_in_

      ##de normalization: 
      predictions = scaler_outputs.inverse_transform(preds)
      model.save(f"./models/rs/random_search_{idx}.h5")
      with open(f"./models/rs/random_search_{idx}.txt", "w") as text_file:
        loss, mse = model.evaluate(testX, testY)
        val_loss, mse = model.evaluate(valX, valY)
        model.summary(print_fn=lambda x: text_file.write(x + '\n'))
        text_file.write(f"\n\ntest loss: {loss} \n")
        text_file.write(f"val loss: {val_loss} \n")
      
      fig, axs = plt.subplots(3, 1, figsize=(10,15), dpi=200)
      for p in predictions:  
        ns = p[::3]
        vs = p[1::3]
        ts = p[2::3]
        
        axs[0].plot(ns)
        axs[1].plot(vs)
        axs[2].plot(ts)
        # for i in range(3):
          # axs[i].plot(p[i*640:(i+1)*640])
      axs[0].set_yscale("log")
      axs[1].set_yscale("log")
      axs[2].set_yscale("log")
        
      plt.savefig(f"./models/rs/random_search_{idx}.png")
      plt.close(fig)
      
      #Normalize the inputs
      #predictions
      bla = model.predict(valX)
      preds = pd.DataFrame(bla)
      preds.columns = scaler_outputs.feature_names_in_

      ##de normalization: 
      predictions = scaler_outputs.inverse_transform(preds)
      
      fig, axs = plt.subplots(3, 1, figsize=(10,15), dpi=200)
      for p in predictions:  
        ns = p[::3]
        vs = p[1::3]
        ts = p[2::3]
        
        axs[0].plot(ns)
        axs[1].plot(vs)
        axs[2].plot(ts)
        # for i in range(3):
          # axs[i].plot(p[i*640:(i+1)*640])
      axs[0].set_yscale("log")
      axs[1].set_yscale("log")
      axs[2].set_yscale("log")
        
      plt.savefig(f"./models/rs/val_random_search_{idx}.png")
      plt.close(fig)
      
      
      # pd.DataFrame(predictions).to_csv("./out/haha/random_search.csv")
      
    # predictions = best_model_r.predict(testX)
    # print(predictions[:5])
    # print(predictions.shape)
    
    # plot_data_values(predictions, "HAHHA", figsize=(10,10), dpi=200)
    
    # plt.savefig("./out/haha/random_search.png")
    
    
    
    