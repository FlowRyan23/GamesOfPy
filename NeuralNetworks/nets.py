from keras import Model, Sequential
from keras.layers import Dense
from keras.optimizers import Adam, Optimizer
from keras.metrics import mse

DEFAULT_TB_LOG_DIR = "./NeuralNetworks/logs/"


def fully_connected(shape: list, activation: str = "relu", optimizer: Optimizer = Adam, loss: str = "mean_squared_error", metrics: list = mse) -> Model:
	model = Sequential()
	for units, i in enumerate(shape):
		if i == 0:
			continue
		elif i == 1:
			model.add(Dense(units, activation=activation, input_dim=shape[0]))
		else:
			model.add(Dense(units, activation=activation))

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	return model
