import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


sciezka_pliku = "dane_ekonomiczne.csv"

columns = ["Close_USDPLN", "Close_EURPLN", "Close_EURUSD", "Close_EURGBP"]

look_back = 40


# fix random seed for reproducibility
tf.random.set_seed(7)

dataframe = pd.read_csv(sciezka_pliku, usecols=["Close_EURPLN"], engine='python')
print(dataframe.head())

# plt.plot(dataset)
# plt.show()

#zmiana na logarytmiczne stopy zwrotu
# log(Cena_t / Cena_{t-1})
dataframe['Log_Returns'] = np.log(dataframe['Close_EURPLN'] / dataframe['Close_EURPLN'].shift(1))

# Usuwamy pierwszy wiersz, który po przesunięciu będzie miał wartość NaN
df_returns = dataframe[['Log_Returns']].dropna()



dataset = df_returns.values
dataset = dataset.astype('float32')

#podzielanie danych na zbior testory i treningowy
train_size = int(len(dataset) * 0.8)

train = dataset[:train_size]
test = dataset[train_size:]

#normalizing
scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
test = scaler.transform(test) 


print(len(train),len(test))


#zmien dane wejsciowe by byly zmianą ceny a nie stałą !!!!


#tworzenie datasetów z lookbackiem
def create_dataset(dataset, lookback=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - lookback-1):
        a = dataset[i:(i+lookback), 0]
        dataX.append(a)
        dataY.append(dataset[i + lookback, 0])
    return np.array(dataX), np.array(dataY)


#przygotowanie datasetów dla modelowania
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(trainX)

# reshape [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back))) #liczba neuronów/komurek pamięci LSTM, (timesteps, features)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2) #epochs - przechodzenie przez dataser, batch_size - aktualizacja wag po przykladzie, verbose - ilość logów



# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# 1. Przygotowanie pustych tablic do rysowania (tej samej wielkości co oryginał)
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# Wstawiamy predykcje treningowe w odpowiednie miejsce
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# 2. Przygotowanie tablicy dla testów
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# Wyliczamy start: po treningu + margines look_back
test_start = len(trainPredict) + (look_back * 2) + 1
testPredictPlot[test_start:test_start + len(testPredict), :] = testPredict

# 3. Jeden wspólny wykres
plt.figure(figsize=(15, 8))

# Dane rzeczywiste (zawsze pod spodem)
plt.plot(scaler.inverse_transform(dataset), label='Dane Rzeczywiste', color='blue', alpha=0.3)

# Predykcje treningowe
plt.plot(trainPredictPlot, label='Predykcja: Zbiór Treningowy', color='orange')

# Predykcje testowe (to co nas najbardziej interesuje)
plt.plot(testPredictPlot, label='Predykcja: Zbiór Testowy', color='green')

plt.title('Porównanie Kursu Rzeczywistego z Predykcjami LSTM')
plt.xlabel('Dni / Obserwacje')
plt.ylabel('Cena (PLN)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()