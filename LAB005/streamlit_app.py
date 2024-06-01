import pandas as pd
import streamlit as st
import joblib
import time

# Ścieżka do wcześniej wytrenowanego modelu
model = joblib.load('LAB005/model_lreg.joblib')

sex_d = {0: 'Kobieta', 1: 'Mężczyzna'}
pclass_d = {0: 'Pierwsza', 1: 'Druga', 2: 'Trzecia'}
embarked_d = {0: 'Cherbourg', 1: 'Queenstown', 2: 'Southampton'}


def main():
    st.set_page_config(page_title='Titanic')

    # Wyświetlenie informacji o prawidłowym uruchomieniu aplikacji
    st.success('Aplikacja została prawidłowo uruchomiona')

    # Ładowanie aplikacji
    st.spinner()
    with st.spinner(text='Ładowanie...'):
        time.sleep(2)

    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    with overview:
        st.image(
            'https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG')
        st.header('Czy przeżyłbyś katastrofę?')

    with left:
        sex_radio = st.radio('Płeć', list(sex_d.keys()), format_func=lambda x: sex_d[x])
        pclass_radio = st.radio('Klasa', list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
        embarked_radio = st.radio('Port zaokrętowania', list(embarked_d.keys()), index=2,
                                  format_func=lambda x: embarked_d[x])

    with right:
        age_slider = st.slider('Wiek', value=50, min_value=1, max_value=100)
        sibsp_slider = st.slider('# Liczba rodzeństwa i/lub partnera', min_value=0, max_value=8)
        parch_slider = st.slider('# Liczba rodziców i/lub dzieci', min_value=0, max_value=6)
        fare_slider = st.slider('Cena biletu', min_value=0, max_value=500, step=10)

    # Nazwy zmiennych w tej samej kolejności, w jakiej były użyte podczas trenowania modelu
    feature_names = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'male']

    data = pd.DataFrame(
        [[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]],
        columns=feature_names)

    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.header('Czy dana osoba przeżyje? {0}'.format('Tak' if survival[0] == 1 else 'Nie'))
        st.subheader('Pewność predykcji {0:.2f} %'.format(s_confidence[0][survival][0] * 100))


if __name__ == '__main__':
    main()
