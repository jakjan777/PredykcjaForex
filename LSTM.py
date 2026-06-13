import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

# reproducibility
SEED = 42   #42
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

sliding_window = 60
prediction_horizon = 14  
num_of_epochs = 50
embed_dim = 128  # Musi być zgodne z LSTM(units=64)


sciezka_pliku = "dane_ekonomiczne.csv"
data = pd.read_csv(sciezka_pliku)

print(data.info)
print(data.describe)
print(data.dtypes)
print(data.columns)

# pozostawiamy tylko OHLCV dla USDPLN
data = data.drop(columns=["High_EURPLN", "Low_EURPLN", "Open_EURPLN", "Volume_EURPLN",
                "High_EURUSD", "Low_EURUSD", "Open_EURUSD", "Volume_EURUSD",
                "High_EURGBP", "Low_EURGBP", "Open_EURGBP", "Volume_EURGBP",
                "Volume_USDPLN", "Italy_GDP", "Germany_GDP", "France_GDP",
                "EA_ECB_IR", "Close_EURGBP", "Poland_GDP", "United Kingdom_GDP"])

# konwersja daty i usunięcie weekendów (sobota=5, niedziela=6)
data['Date'] = pd.to_datetime(data['Date'])
data = data[~data['Date'].dt.dayofweek.isin([5, 6])].reset_index(drop=True)

# opcjonalnie: ogranicz zakres dat (odkomentuj i zmień daty)
# data = data[(data['Date'] >= "2022-01-01") & (data['Date'] <= "2023-12-31")].reset_index(drop=True)

close_original = data["Close_USDPLN"].copy()
high_original = data["High_USDPLN"].copy()
low_original = data["Low_USDPLN"].copy()
open_original = data["Open_USDPLN"].copy()

# target z skumulowanego zwrotu (ciągła stopa zwrotu za prediction_horizon dni, w procentach)
target = (close_original.shift(-prediction_horizon) / close_original - 1) * 100
data["target"] = target

returns_raw = close_original.pct_change()

# === cechy z RAW close (price-based, przed pct_change) ===
close_raw = data["Close_USDPLN"].copy()

# momentum
data["mom_3"] = close_raw.pct_change(3)
data["mom_5"] = close_raw.pct_change(5)
data["mom_10"] = close_raw.pct_change(10)

# RSI
delta = close_raw.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
data["RSI"] = 100 - (100 / (1 + rs))

# Bollinger position
ma20 = close_raw.rolling(20).mean()
std20 = close_raw.rolling(20).std()
data["bb_pos"] = (close_raw - ma20) / std20.replace(0, np.nan)

# rolling z-score na close
data["zscore_20"] = (close_raw - close_raw.rolling(20).mean()) / close_raw.rolling(20).std().replace(0, np.nan)

# ATR (volatility) — wymaga oryginalnych close/high/low
tr = np.maximum(
    high_original - low_original,
    np.maximum(
        abs(high_original - close_original.shift()),
        abs(low_original - close_original.shift())
    )
)
data["ATR"] = tr.rolling(14).mean()

# opóźniony kierunek jako cecha (z RAW close pct_change)
close_pct = close_original.pct_change()
for lag in [1, 2, 3, 5]:
    data[f"dir_lag_{lag}"] = np.sign(close_pct.shift(lag))

# === konwersja Close_USDPLN, Close_EURPLN i Close_EURUSD na returns ===
data["Close_USDPLN"] = returns_raw
data["Close_EURPLN"] = data["Close_EURPLN"].pct_change()
data["Close_EURUSD"] = data["Close_EURUSD"].pct_change()

# === cechy z returns (return-based) ===
# volatility rolling
data["usd_volatility"] = data["Close_USDPLN"].rolling(10).std()

# cechy z High/Low (USDPLN) — zabezpieczone przed NaN
data["usd_daily_range"] = ((high_original - low_original) / close_original.replace(0, np.nan)).fillna(0)
data["usd_hl_ratio"] = (high_original / low_original.replace(0, np.nan)).fillna(1)

# === NOWE CECHY ===
# MACD (Moving Average Convergence Divergence)
exp1 = close_raw.ewm(span=12, adjust=False).mean()
exp2 = close_raw.ewm(span=26, adjust=False).mean()
data["MACD"] = exp1 - exp2
data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["MACD_histogram"] = data["MACD"] - data["MACD_signal"]

# Bollinger Bands Width (zamiast tylko position)
data["bb_width"] = (std20 * 2) / ma20.replace(0, np.nan)

# Volume-weighted momentum
if "Volume_USDPLN" in data.columns:
    data["volume_trend"] = data["Close_USDPLN"].rolling(10).std() / data["Close_USDPLN"].rolling(10).mean().replace(0, np.nan)

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

close_original = close_original.loc[data.index]
target_raw = data["target"].values

#przygotowanie dla modelu LSTM
features = data.filter([
    "Close_USDPLN",
    "Close_EURPLN",
    "Close_EURUSD",
    

    # momentum
    "mom_3",
    "mom_5",
    "mom_10",

    # techniczne
    "RSI",
    "ATR",
    "bb_pos",
    "bb_width", 
    "usd_volatility",
    "zscore_20",
    "MACD",  
    "MACD_signal", 
    "MACD_histogram",  

    # cechy z High/Low
    "usd_daily_range",
    "usd_hl_ratio",

    # stopy procentowe
    "US_FED",
    "EA_ECB",
    "PL_NBP",

    # opóźniony kierunek
    "dir_lag_1",
    "dir_lag_2",
    "dir_lag_3",
    "dir_lag_5"
])

dataset = features.values

training_data_len = int(np.ceil(len(dataset) * 0.80))

# Preprocessing stages
scaler = StandardScaler()

# dzielenie zbioru na testowy
train_data = dataset[:training_data_len]
test_data = dataset[training_data_len:]

# target też dzielimy (NIE skalowany)
train_target_price = target_raw[:training_data_len]

# skalowanie zbioru treningowego i testowego
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

X_train, Y_train_price = [], []

# create sliding windows, target z RAW wartości (procentowe)
for i in range(sliding_window, len(scaled_train_data) - prediction_horizon + 1):
    X_train.append(scaled_train_data[i-sliding_window:i, :])
    Y_train_price.append(train_target_price[i - 1])

X_train = np.array(X_train)
Y_train_price = np.array(Y_train_price)

# Klasyfikacja binana
Y_train_binary = (Y_train_price > 0).astype(int)

print(f"Class balance: {np.mean(Y_train_binary)*100:.1f}% UP, {np.mean(Y_train_binary==0)*100:.1f}% DOWN")

# jawny podział train/validation 
gap = sliding_window + prediction_horizon

val_split = int(len(X_train) * 0.9)
X_val, Y_val_binary = X_train[val_split:], Y_train_binary[val_split:]
X_train, Y_train_binary = X_train[:val_split - prediction_horizon], Y_train_binary[:val_split - prediction_horizon]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(test_data) - prediction_horizon + 1}")

# Obliczenie class weights - ważyć niezbalansowane klasy

class_weights = compute_class_weight('balanced', classes=np.unique(Y_train_binary), y=Y_train_binary)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: DOWN={class_weight_dict[0]:.2f}, UP={class_weight_dict[1]:.2f}")

class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.bn = keras.layers.BatchNormalization()
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        #attn_output = self.bn(attn_output, training=training) CHYBA NIEPOTRZEBNE
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

# macierz stałych wartości sinus i cosinus
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        # Generowanie macierzy pozycji
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        
        # Zastosowanie sin dla parzystych i cos dla nieparzystych indeksów
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        # Zapisanie jako stała 
        self.pos_encoding = tf.convert_to_tensor(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        # inputs shape: (batch, seq_len, embed_dim)
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "embed_dim": self.embed_dim})
        return config


# building MODEL — Hybryda LSTM + Transformer
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(128, return_sequences=True)(inputs)  # zwiększone z 32 na 128
x = keras.layers.Dropout(0.5)(x)  # zwiększone z 0.2 na 0.3
x = PositionalEncoding(max_len=sliding_window, embed_dim=embed_dim)(x)  # DODANIE POZYCJI
x = TransformerBlock(embed_dim=128, num_heads=16, ff_dim=256)(x)  # zwiększone wymiary
#x = keras.layers.Flatten()(x)  # zamiast GlobalAveragePooling1D                CHYBA DO WYWALENIA
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x)
x = keras.layers.Dropout(0.5)(x)


price_out = keras.layers.Dense(1, activation='sigmoid', name="price")(x)
model = keras.Model(inputs=inputs, outputs=price_out)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),  
                loss='binary_crossentropy',
                metrics=['accuracy'])
train_targets = Y_train_binary
val_targets = Y_val_binary

model.summary()

# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    mode='max',  
    patience=50,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=0.00001,
    verbose=1
)

training = model.fit(X_train,
                     train_targets,
                     epochs=num_of_epochs, 
                     batch_size=16,
                     callbacks=[early_stop, reduce_lr],
                     validation_data=(X_val, val_targets),
                     class_weight=class_weight_dict,  # ważyć niezbalansowane klasy
                     shuffle=False)

# połączenie końcówki train + test (żeby mieć pełne okno)
test_inputs = np.concatenate(
    (scaled_train_data[-sliding_window:], scaled_test_data),
    axis=0
)

n_test = len(test_data) - prediction_horizon + 1
Y_test_price_raw = target_raw[training_data_len - 1 : training_data_len - 1 + n_test]
Y_test_price_raw = Y_test_price_raw.reshape(-1, 1)

X_test = []
for i in range(sliding_window, len(test_inputs) - prediction_horizon + 1):
    X_test.append(test_inputs[i-sliding_window:i, :])


X_test = np.array(X_test)

# przygotowywanie Y_test (target dla klasyfikacji)
n_test = len(X_test)
# Pobranie binarnego targetów z oryginalnego targetu (0 = spadek, 1 = wzrost)
Y_test_binary = (target_raw[training_data_len - 1 : training_data_len - 1 + n_test] > 0).astype(int)

# Predykcje
predictions = model.predict(X_test)      
training_predictions = model.predict(X_train)

# konwersja sigmoid output (0-1) na klasy (0/1) - prosty threshold 0.5
predictions_lstm = (predictions.flatten() > 0.5).astype(int)
training_predictions_lstm = (training_predictions.flatten() > 0.5).astype(int)


def naive_benchmark(Y_true_dir, Y_pred_dir):
    print("\n" + "="*20 + " BENCHMARK " + "="*20)

    always_long = np.ones_like(Y_true_dir)
    always_short = np.zeros_like(Y_true_dir)

    print(f"LSTM Accuracy:  {accuracy_score(Y_true_dir, Y_pred_dir)*100:.2f}%")
    print(f"Zawsze LONG:    {accuracy_score(Y_true_dir, always_long)*100:.2f}%")
    print(f"Zawsze SHORT:   {accuracy_score(Y_true_dir, always_short)*100:.2f}%")

    if accuracy_score(Y_true_dir, Y_pred_dir) > accuracy_score(Y_true_dir, always_long):
        print("SUKCES: Model lepszy niż zawsze LONG")
    else:
        print("PORAŻKA: Model gorszy niż zawsze LONG")


def plot_direction_signals(data, close_original, training_data_len,
                           Y_test_dir, Y_pred_dir, prediction_horizon):
    test_dates = data['Date'].iloc[training_data_len + prediction_horizon - 1:].values

    plt.figure(figsize=(16, 9))
    plt.plot(data['Date'], close_original, label="USD/PLN", color='black', alpha=0.5, linewidth=1)

    buy_signals = Y_pred_dir.flatten() == 1
    sell_signals = Y_pred_dir.flatten() == 0

    plt.scatter(test_dates[buy_signals], close_original.iloc[training_data_len + prediction_horizon - 1:].values[buy_signals],
                color='green', marker='^', s=30, label='KUP', alpha=0.6)
    plt.scatter(test_dates[sell_signals], close_original.iloc[training_data_len + prediction_horizon - 1:].values[sell_signals],
                color='red', marker='v', s=30, label='SPRZEDAJ', alpha=0.6)

    plt.title(f"USD/PLN - Sygnały LSTM (horyzont: {prediction_horizon} dni)")
    plt.xlabel("Data")
    plt.ylabel("Cena (PLN)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    plt.show()



def show_confusion_matrix(Y_true, Y_pred):
    print("\n" + "="*20 + " MACIERZ BŁĘDU " + "="*20)

    cm = confusion_matrix(Y_true, Y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"TP (trafione wzrosty): {tp}")
    print(f"TN (trafione spadki):  {tn}")
    print(f"FP (fałszywe wzrosty): {fp}")
    print(f"FN (fałszywe spadki):  {fn}")

    print("\n" + "="*20 + " METRYKI KLASYFIKACJI " + "="*20)
    print(f"Accuracy:  {accuracy_score(Y_true, Y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(Y_true, Y_pred, pos_label=1)*100:.2f}%")
    print(f"Recall:    {recall_score(Y_true, Y_pred, pos_label=1)*100:.2f}%")
    print(f"F1 Score:  {f1_score(Y_true, Y_pred, pos_label=1)*100:.2f}%")




# TABELA WYNIKÓW TESTU
p_ceny = close_original.iloc[training_data_len - 1 : training_data_len - 1 + n_test].values
rzeczywista_zmiana_bin = Y_test_binary
przewidywana_zmiana_bin = predictions_lstm.flatten()

tabela = pd.DataFrame({
    "Data": data['Date'].iloc[training_data_len - 1 : training_data_len - 1 + n_test].values,
    "Cena": p_ceny,
    "Rzeczywiste": rzeczywista_zmiana_bin,
    "Prognoza": przewidywana_zmiana_bin,
})
print("\n" + "="*70)
print("   TABELA WYNIKÓW (pierwsze 20 wierszy)")
print("="*70)
print(tabela.head(20).to_string(index=False))
print(f"\nŁącznie wierszy: {len(tabela)}")

print("\n" + "="*70)
print("   OSTATNIE 5 PREDYKCJI")
print("="*70)
ostatnie = tabela.tail(5).copy()
ostatnie["Data"] = ostatnie["Data"].astype(str)
for _, w in ostatnie.iterrows():
    real_dir = "UP" if w['Rzeczywiste'] == 1 else "DOWN"
    pred_dir = "UP" if w['Prognoza'] == 1 else "DOWN"
    match = "✓" if w['Prognoza'] == w['Rzeczywiste'] else "✗"
    print(f"{w['Data']}:  Rzeczywiste={real_dir}  Prognoza={pred_dir}  {match}")

# URUCHOMIENIE METRYK
print("\n" + "="*70)
print("   METRYKI KLASYFIKACJI (KIERUNEK CENY)")
print("="*70)

show_confusion_matrix(Y_test_binary, predictions_lstm)
naive_benchmark(Y_test_binary, predictions_lstm)
plot_direction_signals(data, close_original, training_data_len,
                       Y_test_binary, predictions_lstm, prediction_horizon)


