import customtkinter as ctk
from pandasgui import show

from GuiFunctions import Download_Data, Open_Data
from Charts import Show_Forex_Grid

# Ustawienia wyglądu
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Analiza Forex")
        self.geometry("500x300")

        # Nagłówek
        self.label = ctk.CTkLabel(self, text="Panel Sterowania", font=("Arial", 20))
        self.label.pack(pady=20)

        # Przycisk Pobierania
        self.download_btn = ctk.CTkButton(self, text="Pobierz Nowe Dane", command=self.run_analysis)
        self.download_btn.pack(pady=10)

        # Przycisk Pokazywania Tabeli (używając Twojego pandasgui)
        self.show_btn = ctk.CTkButton(self, text="Otwórz PandasGUI", command=self.open_table)
        self.show_btn.pack(pady=10)

        # Przycisk Pokazywania grafów
        self.show_btn = ctk.CTkButton(self, text="Otwórz Wykresy kursów walut", command=self.open_forex)
        self.show_btn.pack(pady=10)

        # Status
        self.status_label = ctk.CTkLabel(self, text="Status: Gotowy", text_color="gray")
        self.status_label.pack(side="bottom", pady=10)

    def run_analysis(self):
        self.status_label.configure(text="Status: Pobieranie...", text_color="yellow")
        print("Uruchamiam pobieranie...")
        x = Download_Data()
        if x != 0:
            self.status_label.configure(text="Status: Zakończono!", text_color="green")
        else:
            self.status_label.configure(text="Status: Błąd", text_color="red")

    def open_table(self):
        self.status_label.configure(text="Status: Otwieranie Danych", text_color="yellow")
        print("Otwieram tabelę...")
        df_final = Open_Data()
        show(df_final)
        self.status_label = ctk.CTkLabel(self, text="Status: Gotowy", text_color="gray")

    def open_forex(self):
        self.status_label.configure(text="Status: Otwieranie wykresów", text_color="yellow")
        print("Otwieram wykresy...")
        df_final = Open_Data()
        Show_Forex_Grid(df_final)
        self.status_label = ctk.CTkLabel(self, text="Status: Gotowy", text_color="gray")

if __name__ == "__main__":
    app = App()
    app.mainloop()