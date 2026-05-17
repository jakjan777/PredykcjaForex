from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error

sliding_window = 60


sciezka_pliku = "dane_ekonomiczne.csv"

columns = ["Close_USDPLN", "Close_EURPLN", "Close_EURUSD", "Close_EURGBP"]

data = pd.read_csv(sciezka_pliku)

print(data.info)
print(data.describe)
print(data.dtypes)



#wizualizacja danych
def usd_pln_vis():
    plt.figure(figsize=(12,6))
    plt.plot(data['Date'], data['Close_USDPLN'], label="close USD/PLN", color="red")
    plt.title("open-close USD/PLN")
    plt.legend()
    plt.show()


#dropping columns that are not important right now
data = data.drop(columns=["High_USDPLN", "Low_USDPLN", "Open_USDPLN", "Volume_USDPLN",
                          "High_EURPLN", "Low_EURPLN", "Open_EURPLN", "Volume_EURPLN",
                          "High_EURUSD", "Low_EURUSD", "Open_EURUSD", "Volume_EURUSD",
                          "High_EURGBP", "Low_EURGBP", "Open_EURGBP", "Volume_EURGBP"])

#dropping non numeric data (dates)
data_numeric = data.select_dtypes(include=["int64", "float64"])

#sprawdzanie koleracji (heatmap)
def heatmap():
    plt.figure(figsize=(16,9))
    sns.heatmap(data_numeric.corr(), annot=True, cmap="coolwarm")
    plt.title("feature corelation")
    plt.show()
#heatmap()

#konwersja daty na prawidlowy obiekt
data['Date'] = pd.to_datetime(data['Date'])


# ograniczenie przedzialu czasowego
# data = data.loc[
#     (data['Date'] > datetime(2023,1,1)) &
#     (data['Date'] < datetime(2023,12,31))
# ].copy()

#TODO dodac więcej zależności między walutami

#TODO zmiany wahania kursów walut
close_original = data["Close_USDPLN"].copy()
data["Close_USDPLN"] = data["Close_USDPLN"].pct_change()
data["Close_EURPLN"] = data["Close_EURPLN"].pct_change()

#TODO dodac do modelu
#rolling mean
data["usd_ma_5"] = data["Close_USDPLN"].rolling(5).mean()
data["usd_ma_10"] = data["Close_USDPLN"].rolling(10).mean()

#volatility
data["usd_volatility"] = data["Close_USDPLN"].rolling(10).std()

#momentum
data["usd_momentum"] = data["Close_USDPLN"] - data["Close_USDPLN"].shift(5)

#spready
data["eurusd_usdpln_ratio"] = data["Close_EURUSD"] / data["Close_USDPLN"]

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

close_original = close_original.loc[data.index]

#przygotowanie dla modelu LSTM
# features = data.filter(["Close_USDPLN"])
features = data.filter([
    "Close_USDPLN",
    "Close_EURPLN",
    "Close_EURUSD",
    "Close_EURGBP",

    # nowe feature
    "usd_ma_5",
    "usd_ma_10",
    "usd_volatility",
    "usd_momentum",
    "eurusd_usdpln_ratio"
])

dataset = features.values #.reshape(-1, 1) #konwersja do np w wersji 2D

training_data_len = int(np.ceil(len(dataset) * 0.95))      #rozmiar zbioru testowego/treningowego


#Preprocessing stages
scaler = StandardScaler()
# scaler = MinMaxScaler()


#dzielenie zbioru na testowy
train_data = dataset[:training_data_len]
test_data = dataset[training_data_len:]

#skalowanie zbioru treningowego i testowego
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

X_train, Y_train = [], []


#create a sliding windows for our data oraz tworzymy dane wejsciowe
for i in range(sliding_window, len(scaled_train_data)):
    # Pobieramy wszystkie kolumny (USDPLN i EURPLN)
    X_train.append(scaled_train_data[i-sliding_window:i, :]) 
    # Przewidujemy tylko USDPLN (indeks 0)
    Y_train.append(scaled_train_data[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

#tworzenie rozmiaru macierzy // zmieniane w zaleznosci od tego ile parametrow przyjmujemy - ostatniacyfra to liczba cech (features = 2)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], dataset.shape[1]))


#buildin model
model = keras.models.Sequential()

#1 warstwa
model.add(keras.layers.LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# 2 warstwa
model.add(keras.layers.LSTM(32, return_sequences=False))
# 3 warstwa
model.add(keras.layers.Dense(16, activation="relu"))
# 4 warstw
model.add(keras.layers.Dropout(0.2))           
# 5 warstwa (finalowa)
model.add(keras.layers.Dense(1))


model.summary()
model.compile(optimizer="adam", 
              loss="mse", 
              metrics=[
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.MeanSquaredError(),
                keras.metrics.RootMeanSquaredError()
                ]
            )
#TODO: przetestowac funkcje straty loss="mse" lub loss=keras.losses.Huber(), loss="mae"


# Early stopping które ma pomóc zapobiec overfittowaniu
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

training = model.fit(X_train, 
                     Y_train, 
                     epochs=50, 
                     batch_size=32,
                     callbacks=[early_stop],
                     validation_split=0.1,
                     shuffle=False
                     )

# połączenie końcówki train + test
test_inputs = np.concatenate(
    (scaled_train_data[-sliding_window:], scaled_test_data),
    axis=0
)

X_test = []

#to sa wartosci rzeczywiste do porownania
Y_test = dataset[training_data_len:]

for i in range(sliding_window, len(test_inputs)):
    # ZMIANA: bierzemy wszystkie kolumny (:) a nie tylko indeks 0
    X_test.append(test_inputs[i-sliding_window:i, :]) 

X_test = np.array(X_test)
# ZMIANA: ostatni wymiar to dataset.shape[1] (czyli 2)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], dataset.shape[1]))



#making predictions
predictions = model.predict(X_test)
#predictions = scaler.inverse_transform(predictions)
training_predictions = model.predict(X_train)


# Funkcja pomocnicza do odwracania skali dla jednej kolumny
def inverse_transform_only_first(scaler, predictions, n_features):
    # Tworzymy pustą macierz o szerokości oryginalnego datasetu
    temp_mat = np.zeros((len(predictions), n_features))
    # Wstawiamy nasze predykcje w pierwszą kolumnę (USDPLN)
    temp_mat[:, 0] = predictions.flatten()
    # Odwracamy skalowanie i wyciągamy tylko pierwszą kolumnę
    return scaler.inverse_transform(temp_mat)[:, 0].reshape(-1, 1)

# Użycie:
predictions_inv = inverse_transform_only_first(scaler, predictions, dataset.shape[1])
training_predictions_inv = inverse_transform_only_first(scaler, training_predictions, dataset.shape[1])






#obliczanie jak bardzo model jest skuteczny #TODO sprawdz cyz mozna to latwiej zrobic

#training_predictions = scaler.inverse_transform(training_predictions)

# inverse transform targetów
Y_train_inv = inverse_transform_only_first(scaler, Y_train.reshape(-1, 1), dataset.shape[1])
Y_test_inv = Y_test

# RMSE train (używamy zmiennych _inv)
trainScore = np.sqrt(mean_squared_error(Y_train_inv[:, 0], training_predictions_inv[:, 0]))
print('Train Score: %.4f RMSE' % trainScore)

# RMSE test 
testScore = np.sqrt(mean_squared_error(Y_test_inv[:, 0], predictions_inv[:, 0]))
print('Test Score: %.4f RMSE' % testScore)


#NAIWNY TEST CO ON ROBI NIE WIEM ALE MU UFAM
baseline_predictions = Y_test_inv[:-1]

baseline_rmse = np.sqrt(
    mean_squared_error(
        Y_test_inv[1:,0],
        baseline_predictions[:,0]
    )
)

print("Baseline RMSE:", baseline_rmse)

#DIRECTION ACCURACY
pred_dir = np.sign(predictions_inv[1:] - predictions_inv[:-1])
true_dir = np.sign(Y_test_inv[1:,0] - Y_test_inv[:-1,0])

direction_acc = np.mean(pred_dir == true_dir)

print("Direction Accuracy:", direction_acc)



# rekonstrukcja cen z prognoz pct_change
train_dates = data['Date'].iloc[:training_data_len]
test_dates = data['Date'].iloc[training_data_len:]

actual_train_prices = close_original.iloc[:training_data_len]
actual_test_prices = close_original.iloc[training_data_len:]

# rekonstrukcja 1-krokowa: każda predykcja z rzeczywistej poprzedniej ceny
actual_prev = close_original.iloc[training_data_len - 1 : -1].values
predicted_prices = actual_prev * (1 + predictions_inv.flatten())

plt.figure(figsize=(16,9))

plt.plot(train_dates, actual_train_prices,
         label="Train (actual)", color='red')

plt.plot(test_dates, actual_test_prices,
         label="Test (Actual)", color='green')

plt.plot(test_dates, predicted_prices,
         label="Predictions", color='blue')

plt.title("USD/PLN market prices")
plt.xlabel("Date")
plt.ylabel("close price")
plt.legend()
plt.show()
