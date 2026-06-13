import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, accuracy_score)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

sliding_window = 90
prediction_horizon = 5
transaction_cost = 0.0005
threshold = 0.3
MC_SAMPLES = 30

sciezka_pliku = "dane_ekonomiczne.csv"
data = pd.read_csv(sciezka_pliku)

data = data.drop(columns=["High_EURPLN", "Low_EURPLN", "Open_EURPLN", "Volume_EURPLN",
                          "High_EURUSD", "Low_EURUSD", "Open_EURUSD", "Volume_EURUSD",
                          "High_EURGBP", "Low_EURGBP", "Open_EURGBP", "Volume_EURGBP"])

data['Date'] = pd.to_datetime(data['Date'])
data = data[~data['Date'].dt.dayofweek.isin([5, 6])].reset_index(drop=True)

close_original = data["Close_USDPLN"].copy()
high_original = data["High_USDPLN"].copy()
low_original = data["Low_USDPLN"].copy()
volume_original = data["Volume_USDPLN"].copy()

# target klasyfikacji z progiem
forward_ret = (close_original.shift(-prediction_horizon) / close_original - 1) * 100
target_class = np.where(forward_ret > threshold, 1,
                np.where(forward_ret < -threshold, 0, np.nan))
data["target"] = target_class

returns_raw = close_original.pct_change()

# cechy
close_raw = data["Close_USDPLN"].copy()
data["mom_3"] = close_raw.pct_change(3)
data["mom_5"] = close_raw.pct_change(5)
data["mom_10"] = close_raw.pct_change(10)

delta = close_raw.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
data["RSI"] = 100 - (100 / (1 + rs))
data["bb_pos"] = (close_raw - close_raw.rolling(20).mean()) / close_raw.rolling(20).std().replace(0, np.nan)
data["zscore_20"] = (close_raw - close_raw.rolling(20).mean()) / close_raw.rolling(20).std().replace(0, np.nan)

tr = np.maximum(high_original - low_original,
                np.maximum(abs(high_original - close_original.shift()),
                           abs(low_original - close_original.shift())))
data["ATR"] = tr.rolling(14).mean()
close_pct = close_original.pct_change()
for lag in [1, 2, 3, 5]:
    data[f"dir_lag_{lag}"] = np.sign(close_pct.shift(lag))

data["trend_20"] = close_raw / close_raw.rolling(20).mean() - 1
data["realized_vol_20"] = returns_raw.rolling(20).std()
data["vol_regime"] = (data["realized_vol_20"] > data["realized_vol_20"].rolling(100).mean()).astype(int)
data["skew_20"] = returns_raw.rolling(20).skew()

data["Close_USDPLN"] = returns_raw
data["Close_EURPLN"] = data["Close_EURPLN"].pct_change()
data["usd_volatility"] = data["Close_USDPLN"].rolling(10).std()
data["usd_daily_range"] = ((high_original - low_original) / close_original.replace(0, np.nan)).fillna(0)
data["usd_hl_ratio"] = (high_original / low_original.replace(0, np.nan)).fillna(1)

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
close_original = close_original.loc[data.index]

target_raw = data["target"].values
features = data.filter([
    "Close_USDPLN", "Close_EURPLN", "Close_EURUSD", "Close_EURGBP",
    "mom_3", "mom_5", "mom_10",
    "RSI", "ATR", "bb_pos", "usd_volatility", "zscore_20",
    "trend_20", "realized_vol_20", "vol_regime", "skew_20",
    "usd_daily_range", "usd_hl_ratio",
    "US_FED", "EA_ECB", "PL_NBP",
    "dir_lag_1", "dir_lag_2", "dir_lag_3", "dir_lag_5"
])

dataset = features.values
valid = ~np.isnan(target_raw)
dataset = dataset[valid]
target_raw = target_raw[valid].astype(int)
close_original = close_original.loc[data.index[valid]]
data = data.loc[data.index[valid]].reset_index(drop=True)
close_original = close_original.reset_index(drop=True)

print(f"Class balance: {np.mean(target_raw==1)*100:.1f}% UP, {np.mean(target_raw==0)*100:.1f}% DOWN")


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.08, **kwargs):
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
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                       "ff_dim": self.ff_dim, "rate": self.rate})
        return config


def build_model(input_shape, dropout_rate=0.08):
    inp = keras.layers.Input(shape=input_shape)
    x = keras.layers.LSTM(64, return_sequences=True,
                           recurrent_dropout=dropout_rate,
                           kernel_regularizer=keras.regularizers.l2(1e-4))(inp)
    x = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128, rate=dropout_rate)(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(dropout_rate * 2)(x)  # tylko jeden dropout przed wyjsciem
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


# WALK-FORWARD
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

all_mc_mean, all_mc_std = [], []
all_true_classes, all_actual_ret = [], []
all_dates, all_p, all_p5 = [], [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(dataset)):
    print(f"\n{'='*50}")
    print(f"   FOLD {fold+1}/{n_splits}")
    print(f"   Train: {train_idx[0]}-{train_idx[-1]} Test: {test_idx[0]}-{test_idx[-1]}")
    print(f"{'='*50}")

    train_data = dataset[train_idx]
    test_data = dataset[test_idx]
    train_tgt = target_raw[train_idx]
    test_tgt = target_raw[test_idx]

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)

    X_tr, Y_tr = [], []
    for i in range(sliding_window, len(scaled_train) - prediction_horizon + 1):
        X_tr.append(scaled_train[i-sliding_window:i, :])
        Y_tr.append(train_tgt[i - 1])
    X_tr, Y_tr = map(np.array, (X_tr, Y_tr))
    if len(X_tr) == 0: continue

    print(f"Train: UP={np.mean(Y_tr==1)*100:.1f}%")

    vs = int(len(X_tr) * 0.9)
    X_val, Y_val = X_tr[vs:], Y_tr[vs:]
    X_tr, Y_tr = X_tr[:vs], Y_tr[:vs]

    test_inputs = np.concatenate([scaled_train[-sliding_window:], scaled_test], axis=0)
    n_test = len(test_idx) - prediction_horizon + 1
    idx0 = train_idx[-1]

    X_te = []
    for i in range(sliding_window, len(test_inputs) - prediction_horizon + 1):
        X_te.append(test_inputs[i-sliding_window:i, :])
    X_te = np.array(X_te)
    print(f"Windows - Train: {len(X_tr)} Val: {len(X_val)} Test: {len(X_te)}")

    keras.utils.set_random_seed(SEED + fold)
    model = build_model((X_tr.shape[1], X_tr.shape[2]))
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001, verbose=0)
    model.fit(X_tr, Y_tr, epochs=100, batch_size=32,
              callbacks=[early_stop, reduce_lr],
              validation_data=(X_val, Y_val), shuffle=False, verbose=0)

    # MC Dropout: N predykcji z training=True (bezposrednie wywolanie)
    mc_probs = np.stack([model(X_te, training=True).numpy().flatten() for _ in range(MC_SAMPLES)], axis=1)
    mc_mean = mc_probs.mean(axis=1)
    mc_std = mc_probs.std(axis=1)

    y_true = target_raw[idx0 : idx0 + n_test]
    y_ret = (close_original.iloc[idx0+prediction_horizon : idx0+n_test+prediction_horizon].values /
             close_original.iloc[idx0 : idx0+n_test].values - 1)

    all_mc_mean.append(mc_mean)
    all_mc_std.append(mc_std)
    all_true_classes.append(y_true)
    all_actual_ret.append(y_ret)
    all_dates.append(data['Date'].iloc[idx0:idx0+n_test].values)
    all_p.append(close_original.iloc[idx0:idx0+n_test].values)
    all_p5.append(close_original.iloc[idx0+prediction_horizon : idx0+n_test+prediction_horizon].values)

# AGGREGATE
prob_mean = np.concatenate(all_mc_mean)
prob_std = np.concatenate(all_mc_std)
y_true_all = np.concatenate(all_true_classes)
y_ret_all = np.concatenate(all_actual_ret)
dt_all = np.concatenate(all_dates)
p_all = np.concatenate(all_p)
p5_all = np.concatenate(all_p5)

# sygnaly: trade tylko gdy mala niepewnosc (MC std < 0.08)
sig_conf = (prob_std < 0.08)
signals = np.where((prob_mean > 0.55) & sig_conf, 1,
          np.where((prob_mean < 0.45) & sig_conf, -1, 0))

# METRYKI
pred_dir = (prob_mean > 0.5).astype(int)
acc = accuracy_score(y_true_all, pred_dir)
always_long = np.mean(y_true_all == 1) * 100
cm = confusion_matrix(y_true_all, pred_dir, labels=[0,1])
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*50}")
print(f"   METRYKI MC DROPOUT CLASSIFICATION")
print(f"{'='*50}")
print(f"Accuracy:           {acc*100:.2f}%")
print(f"Zawsze LONG:        {always_long:.2f}%")
print(f"TP={tp} TN={tn} FP={fp} FN={fn}")

# niepewnosc
avg_std_correct = prob_std[pred_dir == y_true_all].mean()
avg_std_wrong = prob_std[pred_dir != y_true_all].mean()
print(f"\nMC std (trafione):  {avg_std_correct:.4f}")
print(f"MC std (bledne):    {avg_std_wrong:.4f}")
print(f"Odrzucono (std>0.08): {(~sig_conf).sum()} z {len(prob_std)} ({np.mean(~sig_conf)*100:.0f}%)")

# SYMULACJA
print(f"\n{'='*20} SYMULACJA (koszt={transaction_cost}) {'='*20}")
buy_hold = np.cumprod(1 + y_ret_all) - 1
sig_change = np.abs(np.diff(signals, prepend=0))
strat_ret = signals * y_ret_all - sig_change * transaction_cost
cum_strat = np.cumprod(1 + strat_ret) - 1
print(f"Buy & Hold:          {buy_hold[-1]*100:.2f}%")
print(f"Strategia MC:        {cum_strat[-1]*100:.2f}%")
print(f"Trade'ow:            {np.sum(signals!=0)} ({np.mean(signals!=0)*100:.0f}% czasu)")
if cum_strat[-1] > buy_hold[-1]:
    print("SUKCES: Model pokonal Buy & Hold")

# przedzial niepewnosci z MC
p10_prob = np.percentile(np.concatenate(all_mc_mean).reshape(-1,1), 10, axis=1)
p90_prob = np.percentile(np.concatenate(all_mc_mean).reshape(-1,1), 90, axis=1)

# ceny: uzywamy progu do konwersji prawdopodobienstwa na cene
p50_price = p_all * np.where(prob_mean > 0.5, 1 + threshold/100, 1 - threshold/100)
p10_price = p_all * np.where(p10_prob > 0.5, 1 + threshold/100, 1 - threshold/100)
p90_price = p_all * np.where(p90_prob > 0.5, 1 + threshold/100, 1 - threshold/100)

tabela = pd.DataFrame({
    "Data": dt_all,
    "Cena dzis": np.round(p_all, 4),
    "P50 (cena)": np.round(p50_price, 4),
    "Rzecz. za 5d": np.round(p5_all, 4),
    "P(UP)": np.round(prob_mean, 3),
    "MC std": np.round(prob_std, 3),
    "Kierunek": np.where(signals == 1, "LONG", np.where(signals == -1, "SHORT", "---")),
})
print("\n" + "="*120)
print("   TABELA (pierwsze 15) - MC DROPOUT")
print("="*120)
print(tabela.head(15).to_string(index=False))
print(f"\nLacznie: {len(tabela)}")

# WYKRES z przedzialem niepewnosci
plt.figure(figsize=(16, 9))
dates_s = pd.to_datetime(dt_all)
idx_s = np.argsort(dates_s)
plt.plot(data['Date'], close_original, color='black', alpha=0.3, linewidth=1, label='USD/PLN')
plt.fill_between(dates_s[idx_s], p10_price[idx_s], p90_price[idx_s],
                 alpha=0.15, color='blue', label='Przedzial niepewnosci (P10-P90)')
plt.plot(dates_s[idx_s], p50_price[idx_s], color='blue', linewidth=1, label='Mediana (P50)')
plt.scatter(dates_s[idx_s], p5_all[idx_s], color='green', s=10, alpha=0.3, label='Rzeczywista')
plt.title("USD/PLN - MC Dropout Classification")
plt.xlabel("Data"); plt.ylabel("Cena (PLN)")
plt.legend(); plt.grid(True, linestyle='--', alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gcf().autofmt_xdate()
plt.show()
