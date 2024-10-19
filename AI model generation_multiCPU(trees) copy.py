import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm  # Dodajemy tqdm do monitorowania postępu
import numpy as np

# Parametry - możesz je zmieniać
model_path = 'C:/Users/adamk/Desktop/Valeo/Yolo model/BRANSON/files/BRANSON_AI_MODEL'
data_path = 'C:/Users/adamk/Desktop/Valeo/Yolo model/BRANSON/files/data'
test_data_path = 'C:/Users/adamk/Desktop/Valeo/Yolo model/BRANSON/files/test'
n_estimators = 100  # Liczba drzew w Random Forest
test_size = 0.3  # Procent danych do testowania
random_state = 42  # Dla powtarzalności wyników
X = np.arange(5000)
y = np.arange(5000)
# Sprawdzenie, czy istnieje zapisany model
model_exists = os.path.exists(model_path)

# Funkcja do wyświetlania menu z kolorowaniem tekstu
def display_menu():
    print("\nWybierz akcję:")
    print("a) Generuj nowy model")
    if model_exists:
        print("\033[92mb) Doucz istniejący model\033[0m")  # Zielony tekst
        print("\033[92mc) Użyj istniejącego modelu do analizy wyników\033[0m")  # Zielony tekst
    else:
        print("\033[90mb) Doucz istniejący model (niedostępne, brak modelu)\033[0m")  # Szary tekst
        print("\033[90mc) Użyj istniejącego modelu do analizy wyników (niedostępne, brak modelu)\033[0m")  # Szary tekst

# Funkcja do sprawdzania unikalnych wartości w kolumnach
def check_column_types(data):
    for column in data.columns:
        unique_values = data[column].unique()
        print(f"Kolumna: {column}, Unikalne wartości: {unique_values}")

# Funkcja do konwersji wartości tekstowych na liczby
def convert_values_to_numeric(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            print(f"Konwertuję wartości w kolumnie: {column}")
            unique_values = data[column].unique()
            print(f"Unikalne wartości: {unique_values}")
            # Konwersja na liczby dla każdej kolumny z danymi tekstowymi
            data[column] = data[column].apply(lambda x: 1 if x in ['OK', 'flat'] else 
                                                (0 if x in ['not flat', 'Leak Detected', 'Maintenance Needed'] else x))
    return data

# Funkcja do generowania nowego modelu
def generate_new_model():
    print("\nGenerowanie nowego modelu...\n")
    
    # Załaduj dane z plików CSV
    print("Ładowanie danych...")
    injection_data = pd.read_csv(os.path.join(data_path, 'injection_traceability.csv'))
    robot_data = pd.read_csv(os.path.join(data_path, 'robot_traceability.csv'))

    # Sprawdź kolumny robot_data i wyświetl unikalne wartości
    check_column_types(robot_data)

    # Konwersja wartości tekstowych na liczby
    print("Konwersja danych na liczby...")
    robot_data = convert_values_to_numeric(robot_data)
    injection_data = convert_values_to_numeric(injection_data)

    # Połączenie danych z wtryskarki i robota na podstawie Cycle ID
    print("Łączenie danych z wtryskarki i robota...")
    data = pd.merge(injection_data, robot_data, on="Cycle ID")

    # Przygotowanie zmiennych X (cechy) i y (target - czy element upadł)
    X = data.drop(columns=["Cycle ID", "Element Dropped"])  # Wszystkie kolumny poza Cycle ID i targetem
    y = data["Element Dropped"]  # Target - czy element upadł (True/False)

    # Podział na dane treningowe i testowe
    print("Podział na dane treningowe i testowe...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Stworzenie modelu Random Forest z włączoną wielowątkowością (n_jobs=-1)
    print("Tworzenie modelu Random Forest...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    # Trenowanie modelu z monitorowaniem postępu
    print("Rozpoczęcie trenowania modelu...")
    for i in tqdm(range(n_estimators)):
        model.fit(X_train, y_train)

    # Zapis modelu do pliku
    print("Zapis modelu do pliku...")
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print("Nowy model został wygenerowany i zapisany jako 'BRANSON_AI_MODEL'.")
    print("Raport z klasyfikacji dla danych testowych:\n")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Funkcja do douczania istniejącego modelu
def retrain_existing_model():
    print("\nDouczanie istniejącego modelu...\n")
    
    # Załaduj dane z plików CSV
    injection_data = pd.read_csv(os.path.join(data_path, 'injection_traceability.csv'))
    robot_data = pd.read_csv(os.path.join(data_path, 'robot_traceability.csv'))

    # Konwersja wartości tekstowych na liczby
    robot_data = convert_values_to_numeric(robot_data)
    injection_data = convert_values_to_numeric(injection_data)  # Konwersja także dla wtryskarki

    # Połączenie danych z wtryskarki i robota na podstawie Cycle ID
    data = pd.merge(injection_data, robot_data, on="Cycle ID")

    # Przygotowanie zmiennych X (cechy) i y (target - czy element upadł)
    X = data.drop(columns=["Cycle ID", "Element Dropped"])  # Wszystkie kolumny poza Cycle ID i targetem
    y = data["Element Dropped"]  # Target - czy element upadł (True/False)

    # Podział na dane treningowe i testowe
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    split_index = int(0.2 * len(X))

    # podział danych

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]





    # Załaduj istniejący model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Dalsze trenowanie modelu
    model.fit(X_train, y_train)

    # Nadpisanie istniejącego modelu
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print("Model został douczony i zapisany ponownie.")
    print("Raport z klasyfikacji dla danych testowych:\n")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Funkcja do używania istniejącego modelu do analizy nowych danych
def use_existing_model():
    print("\nUżywanie istniejącego modelu do analizy wyników...\n")

    # Załaduj dane testowe (np. nowe dane z testów produkcji)
    test_data = pd.read_csv(os.path.join(test_data_path, 'test_data.csv'))

    # Załaduj istniejący model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Przewidywanie na danych testowych
    X_test = test_data.drop(columns=["Cycle ID", "Element Dropped"])  # Wszystkie kolumny poza Cycle ID i targetem
    y_test = test_data["Element Dropped"]  # Target - czy element upadł (True/False)

    y_pred = model.predict(X_test)

    # Wyświetlanie raportu z klasyfikacji
    print("Raport z klasyfikacji dla nowych danych:\n")
    print(classification_report(y_test, y_pred))

# Wyświetlenie menu
display_menu()

# Wybór użytkownika
choice = input("\nWybierz opcję (a, b, c): ").lower()

# Obsługa wyboru użytkownika
if choice == 'a':
    generate_new_model()
elif choice == 'b':
    if model_exists:
        retrain_existing_model()
    else:
        print("\nOpcja douczenia modelu jest niedostępna, ponieważ nie ma istniejącego modelu.")
elif choice == 'c':
    if model_exists:
        use_existing_model()
    else:
        print("\nOpcja analizy modelu jest niedostępna, ponieważ nie ma istniejącego modelu.")
else:
    print("\nNieprawidłowy wybór. Proszę wybrać 'a', 'b' lub 'c'.")
