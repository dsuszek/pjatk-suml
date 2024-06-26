import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

def predict_value(x):
    # Załaduj model z pliku pickle
    with open('our_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Przewiduj wartość y na podstawie dostarczonego x
    y_pred = model.predict([[x]])

    # Zaokrąglij wynik do dwóch miejsc po przecinku
    y_pred_rounded = round(float(y_pred[0]), 2)

    return y_pred_rounded


def update_model(x, y):
    # Załaduj istniejące dane z pliku
    data = pd.read_csv('data.csv')

    # Dodaj nowe dane na końcu pliku
    new_data = pd.DataFrame({'x': [x], 'y': y})
    data = pd.concat([data, new_data], ignore_index=True)
    data.to_csv('data.csv', index=False)

    # Przygotuj dane do trenowania modelu
    X = data[['x']]
    y = data['y']

    # Podziel dane na zestaw treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wczytaj istniejący model
    with open('our_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Wytrenuj model na zaktualizowanych danych
    model.fit(X_train, y_train)

    # Zapisz nowy model do pliku pickle
    with open('our_model.pkl', 'wb') as file:
        pickle.dump(model, file)


def main():
    st.title("Aplikacja do aktualizowania modelu na podstawie nowych danych")

    st.write("## Przewidywanie wartości y")
    x_value = st.number_input("Podaj wartość dla x:", value=0.0)

    if st.button("Przewiduj wartość"):
        y_pred = predict_value(x_value)
        st.write(f"Przewidywana wartość y dla x = {x_value} wynosi: {y_pred}")
        update_model(x_value, y_pred)
        st.write("Model został zaktualizowany na podstawie nowych danych.")


if __name__ == "__main__":
    main()
