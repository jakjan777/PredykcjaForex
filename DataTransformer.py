from pandasgui import show
import pandas as pd

def Transform_Forex_Add_Weekends(USDPLN, EURPLN, EURUSD, EURGBP):

    df = USDPLN.join(EURPLN, rsuffix='_EURPLN')
    df = df.join(EURUSD, rsuffix='_EURUSD')
    df = df.join(EURGBP, rsuffix='_EURGBP')

    # spłaszczenie MultiIndex columns
    df.columns = [
        f"{col[0]}_{col[1].replace('=X', '')}"
        for col in df.columns
    ]

    # Date jako normalna kolumna
    df = df.reset_index()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.set_index("Date")

    full_dates = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="D"
    )

    df = df.reindex(full_dates).ffill().reset_index()

    df = df.rename(columns={"index": "Date"})

    #show(df)
    return df

def Transform_CPI_Daily(df_rr):

    mapowanie = {
        'GB': 'Great_Britain', 'DEU': 'Germany', 'FRA': 'France',
        'ITA': 'Italy', 'USA': 'United_States', 'POL': 'Poland'
    }

    df_rr.columns = [f"{mapowanie.get(col, col)}_CPI" for col in df_rr.columns]
    # rozciągnięcie na dni
    df_daily = df_rr.resample("D").ffill()

    # Date jako kolumna
    df_daily = df_daily.reset_index()
    df_daily = df_daily.rename(columns={"DATE": "Date", "index": "Date"})

    #show(df_daily)
    return df_daily

def Transform_GDP_Daily(all_gdp):
    #Składanie listy w jeden DataFrame
    df = pd.concat(all_gdp).reset_index()
    df.columns = ['Kraj', 'Series', 'Rok', 'Wartosc_PKB']
    
    #Tworzenie pełnej daty
    df['Date'] = pd.to_datetime(df['Rok'].astype(str) + '-01-01')
    
    #Zamiana krajów z wierszy na kolumny
    df_wide = df.pivot(index='Date', columns='Kraj', values='Wartosc_PKB')
    
    #Resampling

    df_daily = df_wide.resample('D').ffill()
    df_daily.columns = [f"{col}_GDP" for col in df_daily.columns]

    df_daily = df_daily.reset_index()
    #show(df_daily)
    return df_daily



def Concat_Data(forex, cpi, gdp):
    # 1. Łączymy Forex z CPI po dacie
    df_combined = pd.merge(forex, cpi, on='Date', how='outer')
    
    # 2. Doklejamy PKB (GDP)
    df_final = pd.merge(df_combined, gdp, on='Date', how='outer')
    
    # 3. Sortujemy chronologicznie
    df_final = df_final.sort_values('Date').reset_index(drop=True)
    
    # 4. Usuwamy wiersze, które zawierają JAKIKOLWIEK null (NaN)
    # axis=0 oznacza wiersze, how='any' oznacza: usuń jeśli choć jedna komórka jest pusta
    df_final = df_final.dropna(axis=0, how='any')
    
    # Resetujemy indeks po usunięciu wierszy, żeby numery szły po kolei (0, 1, 2...)
    df_final = df_final.reset_index(drop=True)
    

    #show(df_final)
    return df_final