import pickle
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


CSV_PATH = "dados.csv"
MODEL_PATH = "modelo.pkl"


def main():
    print("Iniciando treino...")

    if not os.path.exists(CSV_PATH):
        print(f"Erro: Arquivo {CSV_PATH} não encontrado. Execute o coleta.py primeiro.")
        return

    df = pd.read_csv(CSV_PATH)

    if "letra" not in df.columns:
        raise ValueError("O CSV precisa ter a coluna 'letra'.")

    X = df.drop(columns=["letra"])
    y = df["letra"]

    # Trava de segurança: Garante que existem pelo menos duas letras diferentes
    if len(y.unique()) < 2:
        print("Aviso: Apenas uma classe detectada. Colete dados de pelo menos DUAS letras diferentes (ex: A e B) no coleta.py.")
        return

    if len(df) < 5:
        print("Aviso: Muito poucas amostras. Tente coletar algumas dezenas de frames para cada letra.")
        return

    # Só aplica o stratify se houver no mínimo 2 amostras de cada letra coletada
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        print("Aviso: Algumas letras têm apenas 1 amostra. Desativando o 'stratify' para evitar erros de divisão.")
        stratify_param = None
    else:
        stratify_param = y

    # X.values e y.values retiram os rótulos do Pandas, evitando erros no main.py
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=stratify_param
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(modelo, f)

    print(f"Modelo treinado com sucesso!")
    print(f"Acurácia: {acc * 100:.2f}%")
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()