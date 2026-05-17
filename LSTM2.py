from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates

sliding_window = 60


sciezka_pliku = "dane_ekonomiczne.csv"

columns = ["Close_USDPLN", "Close_EURPLN", "Close_EURUSD", "Close_EURGBP"]

data = pd.read_csv(sciezka_pliku)
data_raw = data[['Date', 'Close_USDPLN']].copy()

print(data.info)
print(data.describe)
print(data.dtypes)

#TODO dodac seedy i porównać różne treningi


#wizualizacja danych
def usd_pln_vis():
    plt.figure(figsize=(12,6))
    plt.plot(data['Date'], data['Close_USDPLN'], label="close USD/PLN", color="red")
    plt.title("open-close USD/PLN")
    plt.legend()
    plt.show()


#dropping columns that are not important right now
# pozostawiamy tylko OHLCV dla USDPLN (reszta niepotrzebna)
data = data.drop(columns=["High_EURPLN", "Low_EURPLN", "Open_EURPLN", "Volume_EURPLN",
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
high_original = data["High_USDPLN"].copy()
low_original = data["Low_USDPLN"].copy()
volume_original = data["Volume_USDPLN"].copy()
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

# cechy z High/Low/Volume (USDPLN) — zabezpieczone przed NaN
volume_original = volume_original.fillna(0)
data["usd_daily_range"] = ((high_original - low_original) / close_original.replace(0, np.nan)).fillna(0)
data["usd_hl_ratio"] = (high_original / low_original.replace(0, np.nan)).fillna(1)
data["usd_volume_change"] = volume_original.pct_change().fillna(0)
data["usd_volume_ma_5"] = (volume_original.rolling(5, min_periods=1).mean()
                            / volume_original.rolling(20, min_periods=1).mean().replace(0, np.nan)).fillna(1)

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
    "eurusd_usdpln_ratio",

    # cechy z High/Low/Volume
    "usd_daily_range",
    "usd_hl_ratio",
    "usd_volume_change",
    "usd_volume_ma_5"
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


#building MODEL
model = keras.models.Sequential()

model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.Dropout(0.2)) 

model.add(keras.layers.LSTM(64, return_sequences=False))

model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dropout(0.2))            

model.add(keras.layers.Dense(1))


model.summary()
model.compile(optimizer="adam", 
              loss=keras.losses.Huber(delta=1.0), 
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

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,       # Zmniejsz learning rate o połowę, jeśli...
    patience=4,       # ...przez 4 epoki val_loss nie maleje
    min_lr=0.00001,
    verbose=1
)

training = model.fit(X_train, 
                     Y_train, 
                     epochs=100, 
                     batch_size=32,
                     callbacks=[early_stop, reduce_lr],
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

# Bazowe targety (prawdziwe stopy zwrotu) w oryginalnej skali
Y_train_inv = inverse_transform_only_first(scaler, Y_train.reshape(-1,1), dataset.shape[1])
Y_test_inv = Y_test[:, 0].reshape(-1, 1) # Pierwsza kolumna to prawdziwe pct_change() dla USDPLN

# REKONSTRUKCJA PRAWDZIWYCH CEN (PLN)
# Ceny dla zbioru treningowego
actual_train_prices = close_original.values[sliding_window:training_data_len]
pred_train_prices = []
for i in range(len(training_predictions_inv)):
    # Cena(t) = Cena(t-1) * (1 + pred_pct_change)
    prev_price = close_original.values[sliding_window + i - 1]
    pred_train_prices.append(prev_price * (1 + training_predictions_inv[i, 0]))
pred_train_prices = np.array(pred_train_prices).reshape(-1, 1)

# Ceny dla zbioru testowego
actual_test_prices = close_original.values[training_data_len:]
pred_test_prices = []
for i in range(len(predictions_inv)):
    prev_price = close_original.values[training_data_len + i - 1]
    pred_test_prices.append(prev_price * (1 + predictions_inv[i, 0]))
pred_test_prices = np.array(pred_test_prices).reshape(-1, 1)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n" + "="*20 + " METRYKI STATYSTYCZNE " + "="*20)

# Błędy dla stóp zwrotu (szum rynkowy)
print(f"Test pct_change MAE:  {mean_absolute_error(Y_test_inv, predictions_inv):.5f}")

# Błędy dla realnych cen w PLN (Najważniejsze!)
test_rmse_pln = np.sqrt(mean_squared_error(actual_test_prices, pred_test_prices))
test_mae_pln = mean_absolute_error(actual_test_prices, pred_test_prices)
r2 = r2_score(actual_test_prices, pred_test_prices)

print(f"Test Price RMSE:      {test_rmse_pln:.4f} PLN (Średni błąd kwadratowy)")
print(f"Test Price MAE:       {test_mae_pln:.4f} PLN (Średnio o tyle groszy się myli)")
print(f"Test R^2 Score:       {r2:.4f} (Im bliżej 1, tym lepszy model. Poniżej 0 = gorzej niż średnia)")


print("\n" + "="*20 + " TRAFNOŚĆ KIERUNKU " + "="*20)

# Kierunek rzeczywisty vs przewidywany
true_direction = np.sign(Y_test_inv)
pred_direction = np.sign(predictions_inv)

# Trafność (procent poprawnych wskazań góra/dół)
direction_accuracy = np.mean(true_direction == pred_direction) * 100
print(f"Directional Accuracy: {direction_accuracy:.2f}%")


print("\n" + "="*20 + " BENCHMARK (MODEL NAIWNY) " + "="*20)

# Naiwna predykcja ceny: dzisiejsza cena jest predykcją na jutro
naive_prices = close_original.values[training_data_len-1:-1]
naive_rmse = np.sqrt(mean_squared_error(actual_test_prices, naive_prices))

print(f"LSTM Model Price RMSE:   {test_rmse_pln:.4f} PLN")
print(f"Naive Baseline RMSE:     {naive_rmse:.4f} PLN")

if test_rmse_pln < naive_rmse:
    print(f"SUKCES: Twój model LSTM jest lepszy niż najprostsza strategia punktu odniesienia! Jest lepszy o {naive_rmse - test_rmse_pln} PLN")
else:
    print(f"PORAŻKA: Model przegrywa z prostym przesunięciem wykresu o 1 dzień. Zwiększ look_back lub zmień architekturę. Jest gorszy o {test_rmse_pln - naive_rmse} PLN")



# rekonstrukcja cen z prognoz pct_change
actual_prices_clean = close_original
train_dates = data['Date'].iloc[:training_data_len]
test_dates = data['Date'].iloc[training_data_len:]

actual_train_prices = actual_prices_clean.iloc[:training_data_len]
actual_test_prices = actual_prices_clean.iloc[training_data_len:]

# rekonstrukcja 1-krokowa: każda predykcja z rzeczywistej poprzedniej ceny
actual_prev = close_original.iloc[training_data_len - 1 : -1].values
pred_test_prices = actual_prev * (1 + predictions_inv.flatten())

# 4. TWORZENIE WYKRESU
plt.figure(figsize=(16, 9))

#  Dane z pliku od początku do końca
plt.plot(data['Date'], actual_prices_clean,
         label="Dane z pliku (Całość)", color='yellow', alpha=0.6, linewidth=2)

# Wytrenowane dane treningowe
plt.plot(train_dates, actual_train_prices,
         label="Zbiór Treningowy (Actual)", color='green', linewidth=2)

# Dane predykcji testowej 
plt.plot(test_dates, pred_test_prices,
         label="LSTM Predictions (Test)", color='red', linestyle='--', linewidth=2)

# FORMATOWANIE OSI I WYŚWIETLANIE
plt.title("USD/PLN - Porównanie danych rzeczywistych i predykcji LSTM")
plt.xlabel("Data")
plt.ylabel("Cena (PLN)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Bezpieczne formatowanie osi czasu
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gcf().autofmt_xdate()

plt.show()
