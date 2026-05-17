import yfinance as yf
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandasgui import show

from DataDownloader import Forex_Downloader, CPI_Monthly_Downloader, GDP_Downloader
from DataTransformer import Concat_Data
from Charts import Show_Currency_Chart, Show_Forex_Grid

countries = ['GB','DEU', 'FRA', 'ITA', 'USA', 'POL']
dates = '2022:2026'
sciezka_pliku = "dane_ekonomiczne.csv"


if os.path.exists(sciezka_pliku):
    # Jeśli plik istnieje, wczytaj go
    print("Plik istnieje. Wczytuję dane z dysku...")
    df_final = pd.read_csv(sciezka_pliku)
    # Konwersja kolumny Date z powrotem na format daty
    df_final['Date'] = pd.to_datetime(df_final['Date'])
else:
    print("brak danych lub sciezki")
    


#Show_Forex_Grid(df_final)

# Teraz możesz pracować na df_final
show(df_final)