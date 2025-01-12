import os
import pickle
import numpy as np
import pandas as pd4

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Dla ładnego formatowania tabel
# pip install tabulate
from tabulate import tabulate

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

##############################################################################
# 1. Definicje prostych sieci neuronowych (MLP) w PyTorch
##############################################################################
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

##############################################################################
# 2. Dataset / DataLoader PyTorch
##############################################################################
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

##############################################################################
# 3. Funkcje treningowe (KLASYFIKACJA) - MLP w PyTorch
##############################################################################
def train_mlp_classification(model, loader, criterion, optimizer, device="cpu"):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).view(-1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

def eval_mlp_classification(model, loader, device="cpu"):
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).view(-1)
            probs = torch.sigmoid(outputs)
            all_outputs.append(probs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_outputs)
    targets = np.concatenate(all_targets)
    pred_labels = (preds >= 0.5).astype(int)
    return pred_labels, targets

##############################################################################
# 4. Funkcje treningowe (REGRESJA) - MLP w PyTorch
##############################################################################
def train_mlp_regression(model, loader, criterion, optimizer, device="cpu"):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).view(-1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

def eval_mlp_regression(model, loader, device="cpu"):
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).view(-1)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_outputs)
    targets = np.concatenate(all_targets)
    return preds, targets

##############################################################################
# 5. Wczytywanie danych treningowych i zapisywanie ColumnTransformer
##############################################################################
def load_and_preprocess_data(csv_filename):
    df = pd4.read_csv(csv_filename, parse_dates=["ApplicationDate"])

    # Przykładowe kolumny
    cat_features = [
        'EmploymentStatus',
        'EducationLevel',
        'MaritalStatus',
        'HomeOwnershipStatus',
        'LoanPurpose'
    ]
    num_features = [
        'Age','AnnualIncome','CreditScore','Experience','LoanAmount','LoanDuration',
        'NumberOfDependents','MonthlyDebtPayments','CreditCardUtilizationRate',
        'NumberOfOpenCreditLines','NumberOfCreditInquiries','DebtToIncomeRatio',
        'BankruptcyHistory','PreviousLoanDefaults','PaymentHistory',
        'LengthOfCreditHistory','SavingsAccountBalance','CheckingAccountBalance',
        'TotalAssets','TotalLiabilities','MonthlyIncome','UtilityBillsPaymentHistory',
        'JobTenure','NetWorth','BaseInterestRate','InterestRate','MonthlyLoanPayment',
        'TotalDebtToIncomeRatio'
    ]

    # Usuwamy brakujące
    df = df.dropna(subset=cat_features + num_features + ['LoanApproved','RiskScore'])

    X = df[cat_features + num_features].copy()
    y_class = df['LoanApproved'].astype(int).values
    y_reg   = df['RiskScore'].astype(float).values

    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ])
    X_enc = ct.fit_transform(X)
    if not isinstance(X_enc, np.ndarray):
        X_enc = X_enc.toarray()

    return X_enc, y_class, y_reg, ct

def save_column_transformer(ct, filename="column_transformer.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(ct, f)

def load_column_transformer(filename="column_transformer.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

##############################################################################
# 6. Trenowanie 6 modeli i zapisywanie metryk
##############################################################################
def train_all_models(X_enc, y_class, y_reg):
    # Podział
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_enc, y_class, test_size=0.2, random_state=SEED
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_enc, y_reg, test_size=0.2, random_state=SEED
    )

    classification_metrics = []
    regression_metrics = []

    # --- LogisticRegression (klasyfikacja)
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_train_c, y_train_c)
    pred_lr = lr.predict(X_test_c)
    classification_metrics.append({
        "model_name":"LogisticRegression",
        "accuracy": accuracy_score(y_test_c, pred_lr),
        "precision": precision_score(y_test_c, pred_lr),
        "recall": recall_score(y_test_c, pred_lr),
        "f1": f1_score(y_test_c, pred_lr)
    })
    with open("model_logistic_regression.pkl","wb") as f:
        pickle.dump(lr,f)

    # --- DecisionTreeClassifier (klasyfikacja)
    dtc = DecisionTreeClassifier(random_state=SEED)
    dtc.fit(X_train_c, y_train_c)
    pred_dtc = dtc.predict(X_test_c)
    classification_metrics.append({
        "model_name":"DecisionTreeClassifier",
        "accuracy": accuracy_score(y_test_c, pred_dtc),
        "precision": precision_score(y_test_c, pred_dtc),
        "recall": recall_score(y_test_c, pred_dtc),
        "f1": f1_score(y_test_c, pred_dtc)
    })
    with open("model_decision_tree_classifier.pkl","wb") as f:
        pickle.dump(dtc,f)

    # --- MLPClassifier (PyTorch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim_c = X_enc.shape[1]
    mlp_clf = MLPClassifier(input_dim=input_dim_c).to(device)

    train_dataset_c = SimpleDataset(X_train_c, y_train_c)
    test_dataset_c  = SimpleDataset(X_test_c,  y_test_c)
    train_loader_c = DataLoader(train_dataset_c, batch_size=32, shuffle=True)
    test_loader_c  = DataLoader(test_dataset_c,  batch_size=32, shuffle=False)

    criterion_c = nn.BCEWithLogitsLoss()
    optimizer_c = optim.Adam(mlp_clf.parameters(), lr=0.001)
    EPOCHS_C = 10
    for _ in range(EPOCHS_C):
        train_mlp_classification(mlp_clf, train_loader_c, criterion_c, optimizer_c, device)

    pred_mlp_c, targ_mlp_c = eval_mlp_classification(mlp_clf, test_loader_c, device)
    classification_metrics.append({
        "model_name":"MLPClassifier",
        "accuracy": accuracy_score(targ_mlp_c, pred_mlp_c),
        "precision": precision_score(targ_mlp_c, pred_mlp_c),
        "recall": recall_score(targ_mlp_c, pred_mlp_c),
        "f1": f1_score(targ_mlp_c, pred_mlp_c)
    })
    torch.save(mlp_clf.state_dict(), "model_mlp_classifier.pt")

    # --- LinearRegression (regresja)
    linr = LinearRegression()
    linr.fit(X_train_r, y_train_r)
    pred_linr = linr.predict(X_test_r)
    regression_metrics.append({
        "model_name":"LinearRegression",
        "mse": mean_squared_error(y_test_r, pred_linr),
        "r2": r2_score(y_test_r, pred_linr)
    })
    with open("model_linear_regression.pkl","wb") as f:
        pickle.dump(linr,f)

    # --- DecisionTreeRegressor (regresja)
    dtr = DecisionTreeRegressor(random_state=SEED)
    dtr.fit(X_train_r, y_train_r)
    pred_dtr = dtr.predict(X_test_r)
    regression_metrics.append({
        "model_name":"DecisionTreeRegressor",
        "mse": mean_squared_error(y_test_r, pred_dtr),
        "r2": r2_score(y_test_r, pred_dtr)
    })
    with open("model_decision_tree_regressor.pkl","wb") as f:
        pickle.dump(dtr,f)

    # --- MLPRegressor (PyTorch)
    mlp_reg = MLPRegressor(input_dim=input_dim_c).to(device)

    train_dataset_r = SimpleDataset(X_train_r, y_train_r)
    test_dataset_r  = SimpleDataset(X_test_r,  y_test_r)
    train_loader_r = DataLoader(train_dataset_r, batch_size=32, shuffle=True)
    test_loader_r  = DataLoader(test_dataset_r,  batch_size=32, shuffle=False)

    criterion_r = nn.MSELoss()
    optimizer_r = optim.Adam(mlp_reg.parameters(), lr=0.001)
    EPOCHS_R = 10
    for _ in range(EPOCHS_R):
        train_mlp_regression(mlp_reg, train_loader_r, criterion_r, optimizer_r, device)

    pred_mlp_r, targ_mlp_r = eval_mlp_regression(mlp_reg, test_loader_r, device)
    regression_metrics.append({
        "model_name":"MLPRegressor",
        "mse": mean_squared_error(targ_mlp_r, pred_mlp_r),
        "r2": r2_score(targ_mlp_r, pred_mlp_r)
    })
    torch.save(mlp_reg.state_dict(), "model_mlp_regressor.pt")

    # Zapis metryk do CSV
    save_classification_metrics(classification_metrics, "classification_results.csv")
    save_regression_metrics(regression_metrics,       "regression_results.csv")

    print("Trening 6 modeli zakończony.")

def save_classification_metrics(classification_metrics, filename):
    import csv
    fieldnames = ["model_name","accuracy","precision","recall","f1"]
    write_header = not os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in classification_metrics:
            writer.writerow(row)

def save_regression_metrics(regression_metrics, filename):
    import csv
    fieldnames = ["model_name","mse","r2"]
    write_header = not os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in regression_metrics:
            writer.writerow(row)

##############################################################################
# 7. Skalowanie RiskScore => LOW/MEDIUM/HIGH
##############################################################################
def risk_scale(score):
    if score < 30:
        return "LOW RISK"
    elif score < 60:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

##############################################################################
# 8. Ładowanie wszystkich modeli
##############################################################################
def load_all_models(input_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_class = {}
    models_reg = {}

    # Klasyfikacja
    if os.path.isfile("model_logistic_regression.pkl"):
        with open("model_logistic_regression.pkl","rb") as f:
            models_class["LogisticRegression"] = pickle.load(f)

    if os.path.isfile("model_decision_tree_classifier.pkl"):
        with open("model_decision_tree_classifier.pkl","rb") as f:
            models_class["DecisionTreeClassifier"] = pickle.load(f)

    if os.path.isfile("model_mlp_classifier.pt"):
        mlp_clf = MLPClassifier(input_dim=input_dim).to(device)
        mlp_clf.load_state_dict(torch.load("model_mlp_classifier.pt", map_location=device))
        mlp_clf.eval()
        models_class["MLPClassifier"] = mlp_clf

    # Regresja
    if os.path.isfile("model_linear_regression.pkl"):
        with open("model_linear_regression.pkl","rb") as f:
            models_reg["LinearRegression"] = pickle.load(f)

    if os.path.isfile("model_decision_tree_regressor.pkl"):
        with open("model_decision_tree_regressor.pkl","rb") as f:
            models_reg["DecisionTreeRegressor"] = pickle.load(f)

    if os.path.isfile("model_mlp_regressor.pt"):
        mlp_reg = MLPRegressor(input_dim=input_dim).to(device)
        mlp_reg.load_state_dict(torch.load("model_mlp_regressor.pt", map_location=device))
        mlp_reg.eval()
        models_reg["MLPRegressor"] = mlp_reg

    return models_class, models_reg, device

##############################################################################
# 9. Funkcje do generowania i wyświetlania TABEL dla 3 banków
##############################################################################
def create_three_tables(X_enc, models_class, models_reg, device):
    """
    Tworzy trzy oddzielne tabelki:
      - Bank A: LogisticRegression + LinearRegression
      - Bank B: DecisionTreeClassifier + DecisionTreeRegressor
      - Bank C: MLPClassifier + MLPRegressor
    Zwraca trzy listy wierszy, które można potem wyświetlić za pomocą tabulate.
    """
    # Przygotowanie list wierszy
    rows_bank_a = []
    rows_bank_b = []
    rows_bank_c = []

    # Przechodzimy przez każdy wiersz (rekord)
    for i in range(len(X_enc)):
        row_vector = X_enc[i, :].reshape(1, -1)

        # --- Bank A
        loan_lr = None
        risk_linr = None
        risk_linr_label = None

        if "LogisticRegression" in models_class:
            loan_lr = models_class["LogisticRegression"].predict(row_vector)[0]
        if "LinearRegression" in models_reg:
            val_lr = models_reg["LinearRegression"].predict(row_vector)[0]
            risk_linr = f"{val_lr:.2f}"
            risk_linr_label = risk_scale(val_lr)

        rows_bank_a.append([
            i,
            loan_lr,
            risk_linr,
            risk_linr_label
        ])

        # --- Bank B
        loan_dtc = None
        risk_dtr = None
        risk_dtr_label = None

        if "DecisionTreeClassifier" in models_class:
            loan_dtc = models_class["DecisionTreeClassifier"].predict(row_vector)[0]
        if "DecisionTreeRegressor" in models_reg:
            val_dtr = models_reg["DecisionTreeRegressor"].predict(row_vector)[0]
            risk_dtr = f"{val_dtr:.2f}"
            risk_dtr_label = risk_scale(val_dtr)

        rows_bank_b.append([
            i,
            loan_dtc,
            risk_dtr,
            risk_dtr_label
        ])

        # --- Bank C
        loan_mlp = None
        risk_mlp = None
        risk_mlp_label = None

        if "MLPClassifier" in models_class:
            X_tensor = torch.from_numpy(row_vector).float().to(device)
            with torch.no_grad():
                out_mlp = models_class["MLPClassifier"](X_tensor).view(-1)
                prob_mlp = torch.sigmoid(out_mlp)
                loan_mlp = int((prob_mlp >= 0.5).item())

        if "MLPRegressor" in models_reg:
            X_tensor = torch.from_numpy(row_vector).float().to(device)
            with torch.no_grad():
                val_mlp = models_reg["MLPRegressor"](X_tensor).view(-1).item()
            risk_mlp = f"{val_mlp:.2f}"
            risk_mlp_label = risk_scale(val_mlp)

        rows_bank_c.append([
            i,
            loan_mlp,
            risk_mlp,
            risk_mlp_label
        ])

    return rows_bank_a, rows_bank_b, rows_bank_c

def print_three_tables(rows_a, rows_b, rows_c):
    """
    Wyświetla 3 oddzielne tabelki z wynikami.
    """
    headers_a = ["Index", "Loan(LR)", "Risk(LinR)", "RiskLabel(LinR)"]
    headers_b = ["Index", "Loan(DTC)", "Risk(DTR)", "RiskLabel(DTR)"]
    headers_c = ["Index", "Loan(MLP)", "Risk(MLP)", "RiskLabel(MLP)"]

    print("\n--- Bank A --- (LogisticRegression + LinearRegression)")
    print(tabulate(rows_a, headers=headers_a, tablefmt="pretty"))

    print("\n--- Bank B --- (DecisionTreeClassifier + DecisionTreeRegressor)")
    print(tabulate(rows_b, headers=headers_b, tablefmt="pretty"))

    print("\n--- Bank C --- (MLPClassifier + MLPRegressor)")
    print(tabulate(rows_c, headers=headers_c, tablefmt="pretty"))

##############################################################################
# 10. Predykcja z pliku testowego – teraz z 3 tabelami
##############################################################################
def predict_from_test_file(test_filename):
    if not os.path.isfile("column_transformer.pkl"):
        print("Brak column_transformer.pkl – najpierw wczytaj dane treningowe (opcja 1).")
        return

    ct = load_column_transformer("column_transformer.pkl")

    # Wczytujemy plik testowy
    if not os.path.isfile(test_filename):
        print("Plik testowy nie istnieje.")
        return

    df_test = pd4.read_csv(test_filename, parse_dates=["ApplicationDate"])
    X_test_enc = ct.transform(df_test)
    if not isinstance(X_test_enc, np.ndarray):
        X_test_enc = X_test_enc.toarray()

    # Ładujemy modele
    models_class, models_reg, device = load_all_models(X_test_enc.shape[1])

    # Tworzymy 3 zestawy wierszy
    rows_bank_a, rows_bank_b, rows_bank_c = create_three_tables(
        X_test_enc, models_class, models_reg, device
    )
    # Wyświetlamy w 3 osobnych tabelkach
    print_three_tables(rows_bank_a, rows_bank_b, rows_bank_c)

##############################################################################
# 11. Zapisywanie metryk w formie tabel
##############################################################################
def show_metrics():
    # Klasyfikacja
    if os.path.isfile("classification_results.csv"):
        dfc = pd4.read_csv("classification_results.csv")
        print(tabulate(dfc, headers=dfc.columns, tablefmt="pretty"))
    # Regresja
    if os.path.isfile("regression_results.csv"):
        dfr = pd4.read_csv("regression_results.csv")
        print(tabulate(dfr, headers=dfr.columns, tablefmt="pretty"))

##############################################################################
# 12. Weryfikacja klienta (ręcznie lub z pliku CSV) – z 3 osobnymi tabelami
##############################################################################
def verify_client():
    if not os.path.isfile("column_transformer.pkl"):
        print("Brak column_transformer.pkl – najpierw wczytaj dane treningowe (opcja 1).")
        return

    ct = load_column_transformer("column_transformer.pkl")

    print("Wybierz metodę weryfikacji:")
    print("  a) Podaj dane ręcznie (pojedynczy klient)")
    print("  b) Wczytaj z pliku CSV (wiele rekordów)")
    subchoice = input("Wybierz a/b: ").strip().lower()

    # Te same kolumny (dla spójności z danymi treningowymi)
    cat_features = [
        'EmploymentStatus',
        'EducationLevel',
        'MaritalStatus',
        'HomeOwnershipStatus',
        'LoanPurpose'
    ]
    num_features = [
        'Age','AnnualIncome','CreditScore','Experience','LoanAmount','LoanDuration',
        'NumberOfDependents','MonthlyDebtPayments','CreditCardUtilizationRate',
        'NumberOfOpenCreditLines','NumberOfCreditInquiries','DebtToIncomeRatio',
        'BankruptcyHistory','PreviousLoanDefaults','PaymentHistory',
        'LengthOfCreditHistory','SavingsAccountBalance','CheckingAccountBalance',
        'TotalAssets','TotalLiabilities','MonthlyIncome','UtilityBillsPaymentHistory',
        'JobTenure','NetWorth','BaseInterestRate','InterestRate','MonthlyLoanPayment',
        'TotalDebtToIncomeRatio'
    ]

    if subchoice == "a":
        # WPROWADZAMY 1 KLIENTA RĘCZNIE
        user_data = {}
        print("\nPodaj dane klienta (zatwierdzaj klawiszem ENTER).")
        # Cechy kategoryczne:
        for cat in cat_features:
            val = input(f"{cat}: ").strip()
            user_data[cat] = val

        # Cechy numeryczne:
        for num in num_features:
            val = input(f"{num} (wartość liczbowa): ").strip()
            try:
                user_data[num] = float(val)
            except ValueError:
                print(f"Niepoprawna wartość dla {num}. Przyjmuję 0.")
                user_data[num] = 0.0

        # Tworzymy 1-wierszowy DataFrame
        df_user = pd4.DataFrame([user_data])
        X_user_enc = ct.transform(df_user)
        if not isinstance(X_user_enc, np.ndarray):
            X_user_enc = X_user_enc.toarray()

        models_class, models_reg, device = load_all_models(X_user_enc.shape[1])

        # Tworzymy tabele (de facto będzie jedna linijka w każdej tabeli)
        rows_bank_a, rows_bank_b, rows_bank_c = create_three_tables(
            X_user_enc, models_class, models_reg, device
        )
        print_three_tables(rows_bank_a, rows_bank_b, rows_bank_c)

    elif subchoice == "b":
        # WPROWADZAMY DANE Z PLIKU CSV (MOGĄ BYĆ WIELE WIERSZY)
        csv_file = input("Podaj nazwę pliku CSV z danymi klientów: ").strip()
        if not os.path.isfile(csv_file):
            print("Plik nie istnieje.")
            return

        df_csv = pd4.read_csv(csv_file)
        # Zakładamy, że plik CSV ma te same kolumny co dane treningowe
        X_enc_csv = ct.transform(df_csv)
        if not isinstance(X_enc_csv, np.ndarray):
            X_enc_csv = X_enc_csv.toarray()

        models_class, models_reg, device = load_all_models(X_enc_csv.shape[1])

        # Tworzymy trzy oddzielne zestawy wierszy
        rows_bank_a, rows_bank_b, rows_bank_c = create_three_tables(
            X_enc_csv, models_class, models_reg, device
        )
        print_three_tables(rows_bank_a, rows_bank_b, rows_bank_c)

    else:
        print("Nieprawidłowy wybór. Powrót do menu głównego.")

##############################################################################
# 13. MENU GŁÓWNE
##############################################################################
def main_menu():
    print("\n1. Wczytaj dane TRENINGOWE z CSV i zapisz ColumnTransformer")
    print("2. Trenuj 6 modeli (3 klasyfikacja + 3 regresja)")
    print("3. Wczytaj plik TESTOWY i pokaż wyniki (3 oddzielne tabele)")
    print("4. Pokaż porównanie metryk (tabele z plików CSV)")
    print("5. Weryfikuj klienta (ręcznie lub z pliku CSV) – też 3 tabele")
    print("6. Zakończ")

def main():
    data_loaded = False
    X_enc = None
    y_class = None
    y_reg = None

    while True:
        main_menu()
        choice = input().strip()
        if choice == '1':
            fname = input("Podaj nazwę pliku treningowego (np. loan_data.csv): ").strip()
            if not os.path.isfile(fname):
                print("Plik nie istnieje lub zła ścieżka!")
            else:
                X_enc, y_class, y_reg, ct = load_and_preprocess_data(fname)
                save_column_transformer(ct, "column_transformer.pkl")
                data_loaded = True
                print("Dane zostały wczytane i przetworzone.")
        elif choice == '2':
            if not data_loaded:
                print("Najpierw wczytaj dane (opcja 1).")
            else:
                train_all_models(X_enc, y_class, y_reg)
        elif choice == '3':
            test_fname = input("Podaj nazwę pliku testowego (np. loan_test.csv): ").strip()
            predict_from_test_file(test_fname)
        elif choice == '4':
            show_metrics()
        elif choice == '5':
            verify_client()
        elif choice == '6':
            print("Koniec programu. Do zobaczenia!")
            break
        else:
            print("Nieprawidłowa opcja. Spróbuj ponownie.")

if __name__ == "__main__":
    main()
