#TESTOWY DOKUMENT DO SPRAWDZANIA BIBLIOTEK
import pandas_datareader.data as web
import pandas as pd
from pandasgui import show

def CPI_Monthly_Downloader(dates="2022:2026"):

    import pandas as pd
    import pandas_datareader.data as web

    # parsowanie zakresu
    start_year, end_year = dates.split(":")

    original_start = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    # dodatkowy rok do obliczeń YoY
    extended_start = f"{int(start_year)-1}-01-01"

    symbols = {
        'GB':  'GBRCPIALLMINMEI',
        'DEU': 'DEUCPIALLMINMEI',
        'FRA': 'FRACPIALLMINMEI',
        'ITA': 'ITACPIALLMINMEI',
        'USA': 'USACPIALLMINMEI',
        'POL': 'POLCPIALLMINMEI'
    }

    try:
        # pobieranie danych
        df = web.DataReader(
            list(symbols.values()),
            'fred',
            extended_start,
            end_date
        )

        # zmiana nazw kolumn
        df.columns = symbols.keys()

        # inflacja rok do roku
        df_rr = df.pct_change(periods=12, fill_method=None) * 100

        # usunięcie dodatkowego roku
        df_rr = df_rr[df_rr.index >= original_start]



    except Exception as e:
        print(f"Błąd: {e}")
        return pd.DataFrame()

# Test
df_miesieczny = CPI_Monthly_Downloader()
show(df_miesieczny)
print(df_miesieczny.tail)