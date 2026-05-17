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


def Transform_Rates_Daily(df_rates):
    # rozciągnięcie wartości miesięcznych na dni
    df_daily = df_rates.resample("D").ffill()
    df_daily = df_daily.reset_index()
    df_daily = df_daily.rename(columns={df_daily.columns[0]: "Date"})
    return df_daily


def Concat_Data(forex, cpi, gdp, rates=None, start_date=None, end_date=None):
    # 1. Łączymy Forex z CPI po dacie
    df_combined = pd.merge(forex, cpi, on='Date', how='outer')
    
    # 2. Doklejamy PKB (GDP)
    df_final = pd.merge(df_combined, gdp, on='Date', how='outer')

    # 3. Doklejamy stopy procentowe (jeśli podane)
    if rates is not None and not rates.empty:
        df_final = pd.merge(df_final, rates, on='Date', how='outer')
    
    # 4. Sortujemy chronologicznie
    df_final = df_final.sort_values('Date').reset_index(drop=True)
    
    # 5. Rozciągnięcie do pełnego zakresu dat (każdy dzień miesiąca)
    if start_date and end_date:
        full_range = pd.date_range(start=start_date, end=end_date, freq="D")
        df_final = df_final.set_index('Date').reindex(full_range).ffill().reset_index()
        df_final = df_final.rename(columns={"index": "Date"})
    
    # 6. Usuwamy wiersze, które zawierają JAKIKOLWIEK null (NaN)
    df_final = df_final.dropna(axis=0, how='any')
    
    # Resetujemy indeks po usunięciu wierszy
    df_final = df_final.reset_index(drop=True)
    
    #show(df_final)
    return df_final