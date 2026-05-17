import yfinance as yf  # type: ignore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.dates as mdates

# =========================
# Pobranie danych
# =========================

data = yf.download("EURUSD=X", start="2014-01-01")

# Kurs zamknięcia
close_prices = data[['Close']]

# =========================
# Normalizacja danych
# =========================

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# =========================
# Tworzenie sekwencji
# =========================

X = []
y = []

window = 30

for i in range(window, len(scaled_data)):
    X.append(scaled_data[i-window:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# =========================
# Podział trening / test
# =========================

split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# =========================
# Model LSTM
# =========================

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),

    LSTM(50),
    Dropout(0.2),

    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =========================
# Trenowanie
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# =========================
# Predykcja
# =========================

predicted = model.predict(X_test)

# Powrót do normalnych wartości
predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(y_test)

# =========================
# Wykres predykcji
# =========================

plt.figure(figsize=(12, 6))

plt.plot(real, label="Prawdziwe kursy")
plt.plot(predicted, label="Przewidziane kursy")

plt.title("Prognozowanie kursu EUR/USD przy użyciu LSTM")
plt.xlabel("Czas")
plt.ylabel("Kurs EUR/USD")

plt.legend()
plt.grid()

plt.show()

# =========================
# Daty dla zbioru testowego
# =========================

dates = data.index[-len(real):]

# =========================
# Wykres predykcji
# =========================


plt.figure(figsize=(14, 6))

plt.plot(dates, real, label="Prawdziwe kursy")
plt.plot(dates, predicted, label="Przewidziane kursy")

plt.title("Prognozowanie kursu EUR/USD przy użyciu LSTM")
plt.xlabel("Data")
plt.ylabel("Kurs EUR/USD")

plt.legend()
plt.grid()

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()