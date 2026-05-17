import pandas as pd
import os
from pandasgui import show

from DataDownloader import Forex_Downloader, CPI_Monthly_Downloader, GDP_Downloader, Interest_Rates_Downloader
from DataTransformer import Concat_Data


sciezka_pliku = "dane_ekonomiczne.csv"


#TU ZMIEN ZAKRES DAT
def Download_Data(dates = '2020:2024', countries = ['GB','DEU', 'FRA', 'ITA', 'USA', 'POL']):   
    forex = (Forex_Downloader(dates))
    cpi = (CPI_Monthly_Downloader())
    gdp = (GDP_Downloader(countries, dates))
    rates = (Interest_Rates_Downloader())

    lewa, prawa = dates.split(':')
    start_date = f"{lewa}-01-01"
    end_date = f"{prawa}-12-31"

    df_final = Concat_Data(forex, cpi, gdp, rates, start_date, end_date)
    df_final.to_csv(sciezka_pliku, index=False)
    print("Dane zostały zapisane do pliku {sciezka_pliku}")

def Open_Data():
    if os.path.exists(sciezka_pliku):
        # Jeśli plik istnieje, wczytaj go
        print("Plik istnieje. Wczytuję dane z dysku...")
        df_final = pd.read_csv(sciezka_pliku)
        # Konwersja kolumny Date z powrotem na format daty
        df_final['Date'] = pd.to_datetime(df_final['Date'])
        return(df_final)
        
    else:
        print("brak danych lub sciezki")
        return 0

