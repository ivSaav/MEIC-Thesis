from keras_tuner import HyperModel
from keras import layers
from keras.models import Sequential

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
            optimizer='adam', loss='mse', metrics=['mse'],
        )
    
        return model