import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

sliding_window = 60
prediction_horizon = 5

sciezka_pliku = "dane_ekonomiczne.csv"
data = pd.read_csv(sciezka_pliku)

print(data.info)
print(data.describe)
print(data.dtypes)

# pozostawiamy tylko OHLCV dla USDPLN (reszta niepotrzebna)
data = data.drop(columns=["High_EURPLN", "Low_EURPLN", "Open_EURPLN", "Volume_EURPLN",
                          "High_EURUSD", "Low_EURUSD", "Open_EURUSD", "Volume_EURUSD",
                          "High_EURGBP", "Low_EURGBP", "Open_EURGBP", "Volume_EURGBP"])

# konwersja daty i usunięcie weekendów (sobota=5, niedziela=6)
data['Date'] = pd.to_datetime(data['Date'])
data = data[~data['Date'].dt.dayofweek.isin([5, 6])].reset_index(drop=True)

# opcjonalnie: ogranicz zakres dat (odkomentuj i zmień daty)
# data = data[(data['Date'] >= "2022-01-01") & (data['Date'] <= "2023-12-31")].reset_index(drop=True)

close_original = data["Close_USDPLN"].copy()
high_original = data["High_USDPLN"].copy()
low_original = data["Low_USDPLN"].copy()
open_original = data["Open_USDPLN"].copy()
volume_original = data["Volume_USDPLN"].copy()

# target z skumulowanego zwrotu (ciągła stopa zwrotu za prediction_horizon dni, w procentach)
target = (close_original.shift(-prediction_horizon) / close_original - 1) * 100
data["target"] = target

# target dla wolumenu (procentowa zmiana wolumenu za prediction_horizon dni)
volume_has_data = volume_original.nunique() > 1
if volume_has_data:
    volume_target = (volume_original.shift(-prediction_horizon) / volume_original.replace(0, np.nan) - 1) * 100
else:
    print("UWAGA: Volume_USDPLN ma wszystkie wartości 0 — wolumen nie będzie przewidywany")
    volume_target = pd.Series(0.0, index=data.index)

data["volume_target"] = volume_target

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

# === konwersja Close_USDPLN i Close_EURPLN na returns ===
data["Close_USDPLN"] = returns_raw
data["Close_EURPLN"] = data["Close_EURPLN"].pct_change()

# === cechy z returns (return-based) ===
# volatility rolling
data["usd_volatility"] = data["Close_USDPLN"].rolling(10).std()

# cechy z High/Low/Volume (USDPLN) — zabezpieczone przed NaN
volume_original_filled = volume_original.fillna(0)
data["usd_daily_range"] = ((high_original - low_original) / close_original.replace(0, np.nan)).fillna(0)
data["usd_hl_ratio"] = (high_original / low_original.replace(0, np.nan)).fillna(1)
data["usd_volume_change"] = volume_original_filled.pct_change().fillna(0)
data["usd_volume_ma_5"] = (volume_original_filled.rolling(5, min_periods=1).mean()
                            / volume_original_filled.rolling(20, min_periods=1).mean().replace(0, np.nan)).fillna(1)

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

close_original = close_original.loc[data.index]
target_raw = data["target"].values
volume_target_raw = data["volume_target"].values

#przygotowanie dla modelu LSTM
features = data.filter([
    "Close_USDPLN",
    "Close_EURPLN",
    "Close_EURUSD",
    "Close_EURGBP",

    # momentum
    "mom_3",
    "mom_5",
    "mom_10",

    # techniczne
    "RSI",
    "ATR",
    "bb_pos",
    "usd_volatility",
    "zscore_20",

    # cechy z High/Low/Volume
    "usd_daily_range",
    "usd_hl_ratio",
    "usd_volume_change",
    "usd_volume_ma_5",

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

training_data_len = int(np.ceil(len(dataset) * 0.95))

# Preprocessing stages
scaler = StandardScaler()

# dzielenie zbioru na testowy
train_data = dataset[:training_data_len]
test_data = dataset[training_data_len:]

# target też dzielimy (NIE skalowany)
train_target_price = target_raw[:training_data_len]
train_target_volume = volume_target_raw[:training_data_len]

# skalowanie zbioru treningowego i testowego
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

X_train, Y_train_price, Y_train_volume = [], [], []

# create sliding windows, target z RAW wartości (procentowe)
for i in range(sliding_window, len(scaled_train_data) - prediction_horizon + 1):
    X_train.append(scaled_train_data[i-sliding_window:i, :])
    Y_train_price.append(train_target_price[i - 1])
    Y_train_volume.append(train_target_volume[i - 1])

X_train = np.array(X_train)
Y_train_price = np.array(Y_train_price)
Y_train_volume = np.array(Y_train_volume)

print(f"Class balance price: {np.mean(Y_train_price > 0)*100:.1f}% UP, {np.mean(Y_train_price <= 0)*100:.1f}% DOWN")

# jawny podział train/validation (czasowy — bierzemy ostatnie 10% train)
val_split = int(len(X_train) * 0.9)
X_val, Y_val_price, Y_val_volume = X_train[val_split:], Y_train_price[val_split:], Y_train_volume[val_split:]
X_train, Y_train_price, Y_train_volume = X_train[:val_split], Y_train_price[:val_split], Y_train_volume[:val_split]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(test_data) - prediction_horizon + 1}")

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
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
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


# building MODEL — Hybryda LSTM + Transformer
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(32, return_sequences=True)(inputs)
x = keras.layers.Dropout(0.2)(x)
x = TransformerBlock(embed_dim=32, num_heads=2, ff_dim=32)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.2)(x)

if volume_has_data:
    price_out = keras.layers.Dense(1, name="price")(x)
    volume_out = keras.layers.Dense(1, name="volume")(x)
    model = keras.Model(inputs=inputs, outputs=[price_out, volume_out])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss={"price": "mean_squared_error", "volume": "mean_squared_error"},
                  loss_weights={"price": 1.0, "volume": 0.5},
                  metrics={"price": "mean_absolute_error", "volume": "mean_absolute_error"})
    train_targets = {"price": Y_train_price, "volume": Y_train_volume}
    val_targets = {"price": Y_val_price, "volume": Y_val_volume}
else:
    price_out = keras.layers.Dense(1, name="price")(x)
    model = keras.Model(inputs=inputs, outputs=price_out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    train_targets = Y_train_price
    val_targets = Y_val_price

model.summary()

# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
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
                     epochs=100,
                     batch_size=32,
                     callbacks=[early_stop, reduce_lr],
                     validation_data=(X_val, val_targets),
                     shuffle=False)

# połączenie końcówki train + test (żeby mieć pełne okno)
test_inputs = np.concatenate(
    (scaled_train_data[-sliding_window:], scaled_test_data),
    axis=0
)

n_test = len(test_data) - prediction_horizon + 1
Y_test_price = target_raw[training_data_len - 1 : training_data_len - 1 + n_test]
Y_test_price = Y_test_price.reshape(-1, 1)
Y_test_volume = volume_target_raw[training_data_len - 1 : training_data_len - 1 + n_test]
Y_test_volume = Y_test_volume.reshape(-1, 1)

X_test = []
for i in range(sliding_window, len(test_inputs) - prediction_horizon + 1):
    X_test.append(test_inputs[i-sliding_window:i, :])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], dataset.shape[1]))

# predictions — multi-output (jeśli volume) lub single-output
predictions = model.predict(X_test)
if volume_has_data:
    predictions_price = predictions[0]
    predictions_volume = predictions[1]
else:
    predictions_price = predictions
    predictions_volume = np.zeros_like(predictions_price)

training_predictions = model.predict(X_train)
if volume_has_data:
    training_predictions_price = training_predictions[0]
    training_predictions_volume = training_predictions[1]
else:
    training_predictions_price = training_predictions
    training_predictions_volume = np.zeros_like(training_predictions_price)

# kierunek ceny (dla benchmarków klasyfikacyjnych)
predictions_dir = (predictions_price > 0).astype(int)
training_predictions_dir = (training_predictions_price > 0).astype(int)





def simulate_strategy(Y_true_dir, Y_pred_dir, actual_returns):
    print("\n" + "="*20 + " SYMULACJA STRATEGII " + "="*20)

    buy_hold = np.cumprod(1 + actual_returns) - 1

    # konwersja 0/1 na -1/1 do symulacji
    trading_signals = 2 * Y_pred_dir.flatten() - 1
    strat_returns = trading_signals * actual_returns
    cumulative_strat = np.cumprod(1 + strat_returns) - 1

    print(f"Buy & Hold:           {buy_hold[-1]*100:.2f}%")
    print(f"Strategia LSTM:       {cumulative_strat[-1]*100:.2f}%")

    if cumulative_strat[-1] > buy_hold[-1]:
        print("SUKCES: Model pokonał Buy & Hold")
    else:
        print("PORAŻKA: Model nie pokonał Buy & Hold")


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
                color='green', marker='^', s=30, label='KUP (LSTM)', alpha=0.6)
    plt.scatter(test_dates[sell_signals], close_original.iloc[training_data_len + prediction_horizon - 1:].values[sell_signals],
                color='red', marker='v', s=30, label='SPRZEDAJ (LSTM)', alpha=0.6)

    plt.title(f"USD/PLN - Sygnały LSTM (horyzont: {prediction_horizon} dni)")
    plt.xlabel("Data")
    plt.ylabel("Cena (PLN)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    plt.show()


def classification_metrics(Y_true, Y_pred):
    print("\n" + "="*20 + " METRYKI KLASYFIKACJI " + "="*20)
    print(f"Accuracy:  {accuracy_score(Y_true, Y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(Y_true, Y_pred, pos_label=1)*100:.2f}%")
    print(f"Recall:    {recall_score(Y_true, Y_pred, pos_label=1)*100:.2f}%")
    print(f"F1 Score:  {f1_score(Y_true, Y_pred, pos_label=1)*100:.2f}%")


def show_confusion_matrix(Y_true, Y_pred):
    print("\n" + "="*20 + " MACIERZ BŁĘDU " + "="*20)

    cm = confusion_matrix(Y_true, Y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"TP (trafione wzrosty): {tp}")
    print(f"TN (trafione spadki):  {tn}")
    print(f"FP (fałszywe wzrosty): {fp}")
    print(f"FN (fałszywe spadki):  {fn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\nPrecyzja wzrostów: {precision*100:.2f}%")
    print(f"Czułość wzrostów:  {recall*100:.2f}%")


# --- przygotowanie actual_returns do symulacji ---
# forward_ret[t] = zwrot od t do t+prediction_horizon
forward_ret = (close_original.shift(-prediction_horizon) / close_original - 1).values
n_test_samples = len(X_test)
actual_returns_test = forward_ret[training_data_len - 1 : training_data_len - 1 + n_test_samples]

# forward_volume_ret[t] = zmiana wolumenu od t do t+prediction_horizon
if volume_has_data:
    forward_vol = (volume_original.shift(-prediction_horizon) / volume_original.replace(0, np.nan) - 1).values
    actual_volume_test = forward_vol[training_data_len - 1 : training_data_len - 1 + n_test_samples]
else:
    actual_volume_test = np.zeros(n_test_samples)

def regression_metrics(Y_true, Y_pred, nazwa):
    print(f"\n{'='*20} METRYKI REGRESJI: {nazwa} {'='*20}")
    print(f"MSE:  {mean_squared_error(Y_true, Y_pred):.6f}")
    print(f"MAE:  {mean_absolute_error(Y_true, Y_pred):.6f}")
    print(f"R2:   {r2_score(Y_true, Y_pred):.4f}")

def direction_accuracy(Y_true, Y_pred, nazwa):
    acc = accuracy_score((Y_true > 0).astype(int), (Y_pred > 0).astype(int))
    print(f"Direction Accuracy ({nazwa}): {acc*100:.2f}%")

# --- TABELA WYNIKÓW TESTU ---
p_ceny = close_original.iloc[training_data_len - 1 : training_data_len - 1 + n_test].values
p5_ceny = close_original.iloc[training_data_len - 1 + prediction_horizon : training_data_len - 1 + n_test + prediction_horizon].values
przewidziane_p5 = p_ceny * (1 + predictions_price.flatten() / 100)

tabela = pd.DataFrame({
    "Data": data['Date'].iloc[training_data_len - 1 : training_data_len - 1 + n_test].values,
    "Cena dziś": p_ceny,
    "Przewidywane za 5d": przewidziane_p5,
    "Rzeczywiste za 5d": p5_ceny,
    "przewidywana zmiana procentowa": predictions_price.flatten(),
    "rzeczywista zmiana procentowa": Y_test_price.flatten(),
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
    print(f"{w['Data']}:  dziś={w['Cena dziś']:.4f}  "
          f"przewidywane={w['Przewidywane za 5d']:.4f}  "
          f"rzeczywiste={w['Rzeczywiste za 5d']:.4f}  "
          f"prognoza={w['przewidywana zmiana procentowa']:+.2f}%  "
          f"fakt={w['rzeczywista zmiana procentowa']:+.2f}%")

# --- URUCHOMIENIE METRYK ---
Y_test_dir_bin = (Y_test_price > 0).astype(int)

print("\n" + "="*50)
print("   METRYKI CENA (USD/PLN)")
print("="*50)
regression_metrics(Y_test_price, predictions_price, "CENA")
direction_accuracy(Y_test_price, predictions_price, "CENA")

if volume_has_data:
    print("\n" + "="*50)
    print("   METRYKI WOLUMEN")
    print("="*50)
    regression_metrics(Y_test_volume, predictions_volume, "WOLUMEN")
    direction_accuracy(Y_test_volume, predictions_volume, "WOLUMEN")
else:
    print("\n(Brak danych wolumenu — metryki wolumenu pominięte)")

print("\n" + "="*50)
print("   METRYKI KIERUNKU CENY")
print("="*50)
classification_metrics(Y_test_dir_bin, predictions_dir)
show_confusion_matrix(Y_test_dir_bin, predictions_dir)
simulate_strategy(Y_test_dir_bin, predictions_dir, actual_returns_test)
naive_benchmark(Y_test_dir_bin, predictions_dir)
plot_direction_signals(data, close_original, training_data_len,
                       Y_test_dir_bin, predictions_dir, prediction_horizon)
