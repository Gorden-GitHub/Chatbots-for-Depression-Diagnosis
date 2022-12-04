from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score



data = pd.read_csv("12 (2).csv")
#print(data.head(0))
data = data.loc[:,['date','variance','high','low','average','Volume XRP','Volume USDT']]
#print(data.head(5))
#print(data.date)

data = data.set_index('date')
data.index = pd.to_datetime(data.index,unit='ns')
#print(data.index)

aim = 'average'

train_data = data.iloc[0:24,:]
print("train")
print(train_data)
test_data = data.iloc[23:30,:]
print("test")
print(test_data)

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
	fig, ax = plt.subplots(1, figsize=(13, 7))
	ax.plot(line1, label=label1, linewidth=lw)
	ax.plot(line2, label=label2, linewidth=lw)
	ax.set_ylabel('AVERAGE Score', fontsize=14)
	ax.set_title(title, fontsize=16)
	ax.legend(loc='best', fontsize=16);

line_plot(train_data[aim], test_data[aim], 'training', 'test', title='')
plt.show()


def normalise_zero_base(continuous):
	return continuous / continuous.iloc[0] - 1

def normalise_min_max(continuous):
	return (continuous - continuous.min()) / (data.max() - continuous.min())


def extract_window_data(continuous, window_len=5, zero_base=True):
	window_data = []
	for idx in range(len(continuous) - window_len):
		tmp = continuous[idx: (idx + window_len)].copy()
		if zero_base:
			tmp = normalise_zero_base(tmp)
		window_data.append(tmp.values)
	return np.array(window_data)
def prepare_data(continuous, aim, window_len=10, zero_base=True, test_size=0.2):
	X_train = extract_window_data(train_data, window_len, zero_base)
	X_test = extract_window_data(test_data, window_len, zero_base)
	y_train = train_data[aim][window_len:].values
	y_test = test_data[aim][window_len:].values
	if zero_base:
		y_train = y_train / train_data[aim][:-window_len].values - 1
		y_test = y_test / test_data[aim][:-window_len].values - 1

	return train_data, test_data, X_train, X_test, y_train, y_test


def build_lstm_model(input_data, output_size, neurons, activ_func='linear',
					 dropout=0.2, loss='mse', optimizer='adam'):
	model = Sequential()
	model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
	model.add(Dropout(dropout))
	model.add(Dense(units=output_size))
	model.add(Activation(activ_func))

	model.compile(loss=loss, optimizer=optimizer)
	return model
np.random.seed(10)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 5
epochs = 20
batch_size = 4
loss = 'mse'
dropout = 0.24
optimizer = 'adam'
train_data, test_data, X_train, X_test, y_train, y_test = prepare_data(
	data, aim, window_len=window_len, zero_base=zero_base, test_size=test_size)


print(X_train.shape)


model = build_lstm_model(
	X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
	optimizer=optimizer)
modelfit = model.fit(
	X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)



plt.plot(modelfit.history['loss'],'r',linewidth=2, label='Training loss')
plt.plot(modelfit.history['val_loss'], 'g',linewidth=2, label='Validation loss')
plt.title('LSTM Neural Networks - Prediction Model')
plt.xlabel('Epochs numbers')
plt.ylabel('MSE numbers')
plt.show()


targets = test_data[aim][window_len:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)

SCORE_MSE=mean_squared_error(preds, y_test)
print(SCORE_MSE)

r2_score=r2_score(y_test, preds)
print(r2_score*100)

preds = test_data[aim].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
line_plot(targets, preds, 'actual', 'prediction', lw=3)
plt.show()

