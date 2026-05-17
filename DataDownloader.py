import world_bank_data as wb
import yfinance as yf
import pandas as pd
from pandasgui import show
import pandas_datareader.data as web

from DataTransformer import Transform_Forex_Add_Weekends, Transform_CPI_Daily, Transform_GDP_Daily, Transform_Rates_Daily
from DataTransformer import Concat_Data


### Pobranie kursów walut dla wybranych krajów
def Forex_Downloader(dates):

    lewa, _ = dates.split(':')
    dates = f"{lewa}-01-01"

    try:
        USDPLN = pd.DataFrame(yf.download("USDPLN=X" ,start=dates))
        EURPLN = pd.DataFrame(yf.download("EURPLN=X" ,start=dates))
        EURUSD = pd.DataFrame(yf.download("EURUSD=X" ,start=dates))
        EURGBP = pd.DataFrame(yf.download("EURGBP=X" ,start=dates))

        return Transform_Forex_Add_Weekends(USDPLN, EURPLN, EURUSD, EURGBP)

    except Exception as e:
        print(f"Błąd przy pobieraniu GDP: {e}")
        return pd.DataFrame()
    


###Pobieranie CPI dla wybranych krajów
def CPI_Monthly_Downloader(start_date="2020-01-01", end_date="2026-01-01"):

    original_start = start_date

    r, m, d = start_date.split("-")
    r = int(r) - 1

    start_date_extended = f"{r}-{m}-{d}"

    symbols = {
        'GB':  'GBRCPIALLMINMEI',
        'DEU': 'DEUCPIALLMINMEI',
        'FRA': 'FRACPIALLMINMEI',
        'ITA': 'ITACPIALLMINMEI',
        'USA': 'USACPIALLMINMEI',
        'POL': 'POLCPIALLMINMEI'
    }

    try:
        df = web.DataReader(
            list(symbols.values()),
            'fred',
            start_date_extended,
            end_date
        )

        df.columns = symbols.keys()

        df_rr = df.pct_change(periods=12, fill_method=None) * 100

        # usunięcie dodatkowego roku
        df_rr = df_rr[df_rr.index >= original_start]

        return Transform_CPI_Daily(df_rr)

    except Exception as e:
        print(f"Błąd przy pobieraniu danych miesięcznych: {e}")
        return pd.DataFrame()


###Pobieranie stóp procentowych z FRED
def Interest_Rates_Downloader(start_date="2020-01-01", end_date="2026-01-01"):

    original_start = start_date
    r, m, d = start_date.split("-")
    r = int(r) - 1
    start_date_extended = f"{r}-{m}-{d}"

    # Każda seria pobierana osobno (jedna może nie istnieć, reszta działa)
    series_map = {
        'US_FED_IR': 'FEDFUNDS',
        'EA_ECB_IR': 'ECBMRRFR',
        'PL_NBP_IR': 'IRSTCI01PLM156N'
    }

    results = {}
    for name, sid in series_map.items():
        try:
            df = web.DataReader(sid, 'fred', start_date_extended, end_date)
            results[name] = df[sid]
            print(f"Pobrano {name} ({sid})")
        except Exception as e:
            print(f"Ostrzeżenie: nie można pobrać {name} ({sid}): {e}")

    if not results:
        print("Błąd: nie pobrano żadnych stóp procentowych.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df[df.index >= original_start]

    return Transform_Rates_Daily(df)


###Pobieranie PKB dla wybranych krajów
def GDP_Downloader(countries, dates, wskaznik='NY.GDP.MKTP.CD'):
    all_gdp = []

    for c in countries:
        try:
            # Pobieranie danych dla konkretnego kraju
            data = wb.get_series(indicator=wskaznik, country=c, date=dates)
            if not data.empty:
                all_gdp.append(data)
                print(f"Pobrano PKB dla: {c}")
        except Exception as e:
            print(f"Błąd PKB dla {c}: {e}")

    if not all_gdp:
        print("Nie pobrano żadnych danych PKB.")
        return pd.DataFrame()
    else:
        return Transform_GDP_Daily(all_gdp)


