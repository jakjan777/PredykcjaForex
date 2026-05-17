import matplotlib.pyplot as plt

def Show_Currency_Chart(df, column_name, color='blue'):

    if column_name not in df.columns:
        print(f"Błąd: Kolumna {column_name} nie istnieje w danych.")
        return

    plt.figure(figsize=(10, 6))
    
    # Rysujemy wybraną kolumnę
    plt.plot(df['Date'], df[column_name], label=column_name, color=color, linewidth=2)

    # Dynamiczny tytuł na podstawie nazwy kolumny
    plt.title(f'Wykres waluty: {column_name.replace("Close_", "")}')
    plt.xlabel('Data')
    plt.ylabel('Kurs')
    plt.legend()
    plt.grid(True)
    plt.show()


def Show_Forex_Grid(df):
    # Sprawdzenie czy kolumny istnieją
    pairs = ['Close_USDPLN', 'Close_EURPLN', 'Close_EURUSD', 'Close_EURGBP']
    colors = ['blue', 'green', 'red', 'purple']
    
    #Tworzenie siatki: 2 wiersze, 2 kolumny
    #figsize określa rozmiar całego okna
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    
    # spłaszczamy tablicę axes do jednej listy, żeby łatwiej było po niej iterować
    axes_flat = axes.flatten()

    #Pętla rysująca każdy wykres w osobnym kwadracie
    for i, col in enumerate(pairs):
        if col in df.columns:
            axes_flat[i].plot(df['Date'], df[col], color=colors[i], linewidth=1.5)
            axes_flat[i].set_title(f'Kurs {col.replace("Close_", "")}')
            axes_flat[i].grid(True, linestyle='--', alpha=0.7)
            axes_flat[i].tick_params(axis='x', rotation=45) # obracamy daty dla czytelności
        else:
            axes_flat[i].set_visible(False) # ukrywamy puste pole, jeśli brak waluty

    #Automatyczne poprawienie odstępów między wykresami
    plt.tight_layout()
    
    #Wyświetlenie okna
    plt.show()

    