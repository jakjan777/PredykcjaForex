# Predykcja kierunku USD/PLN — LSTM + Transformer

Projekt z przedmiotu *Sztuczna inteligencja* (UTP). Celem jest klasyfikacja binarna kierunku kursu USD/PLN na horyzoncie 14 dni roboczych, z wykorzystaniem hybrydowej sieci LSTM i Transformer.

## Zawartość repozytorium

| Plik / katalog | Opis |
|---|---|
| `LSTM.py` | Główny skrypt — przygotowanie cech, trening modelu, ewaluacja |
| `DataDownloader.py` | Pobieranie danych z Yahoo Finance i World Bank |
| `DataTransformer.py` | Transformacja i scalanie danych dziennych |
| `main_gui.py` | Prosty panel GUI (CustomTkinter) do pobierania i podglądu danych |
| `AnalizaForex.py`, `Charts.py`, `GuiFunctions.py` | Analiza i wizualizacja kursów walutowych |
| `dane_ekonomiczne.csv` | Zbiór danych wejściowych (OHLC, pary walutowe, CPI, GDP, stopy procentowe) |
| `Dokumentacja_Predykcja_Forex_LSTM_Transformer.pdf` | Pełna dokumentacja projektu |
| `usun/` | Archiwalne wersje skryptów i notatników |

## Model

- **Wejście:** okno 60 dni × 24 cechy (OHLC USD/PLN, wskaźniki techniczne, dane makroekonomiczne)
- **Architektura:** LSTM(128) → Dropout → Positional Encoding → Transformer Block (8 głów) → Global Average Pooling → Dense → sigmoid
- **Target:** wzrost (`UP`) lub spadek (`DOWN`) skumulowanego zwrotu za 14 dni
- **Podział danych:** 80% trening / 20% test, z walidacją 10% z zbioru treningowego

## Wymagania

Python 3.10+, m.in.:

```
tensorflow
pandas
numpy
scikit-learn
matplotlib
yfinance
world-bank-data
pandas-datareader
customtkinter
pandasgui
```

## Uruchomienie

**Trening i ewaluacja modelu:**

```bash
python LSTM.py
```

**Pobieranie danych i GUI:**

```bash
python main_gui.py
```

Parametry eksperymentów (okno czasowe, learning rate, dropout, liczba głów attention) można zmieniać bezpośrednio w `LSTM.py`.

## Wyniki (konfiguracja bazowa)

| Metryka | Wartość |
|---|---|
| Accuracy | 64,32% |
| Precision | 63,22% |
| Recall | 83,33% |
| F1 | 71,90% |

Model przewyższa naiwny benchmark „zawsze LONG” (54,77%). Szczegóły eksperymentów E1–E4 (okno, głowy attention, learning rate, dropout) — w dokumentacji PDF.
