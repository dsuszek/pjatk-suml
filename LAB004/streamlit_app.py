# Import bibliotek
import streamlit as st
import time
from transformers import pipeline

# Wyświetlenie informacji o prawidłowym uruchomieniu aplikacji
st.success('Aplikacja została prawidłowo uruchomiona')

# Ładowanie aplikacji
st.spinner()
with st.spinner(text='Ładowanie...'):
    time.sleep(2)

# Tytuł aplikacji
st.title('Streamlit Translator')

st.subheader('Aplikacja przygotowana przy wykorzystaniu biblioteki streamlit.')
st.write('Zawiera dwie funkcjonalności:')
st.write('1. Pozwala na tłumaczenie tekstów z języka angielskiego na język niemiecki')
st.write('2. Pozwala sprawdzać wydźwięk emocjonalny tekstu napisanego w języku angielskim.')

option = st.selectbox(
    'Opcje',
    [
        'Tłumacz z języka angielskiego na język niemiecki',
        'Wydźwięk emocjonalny tekstu (eng)'
    ],
)

if option == 'Tłumacz z języka angielskiego na język niemiecki':
    text_to_translate = st.text_area(label='Proszę podać tekst do przetłumaczenia')
    if st.button('Przetłumacz', type='primary') and text_to_translate:
        try:
            translator = pipeline('translation_en_to_de', model='Helsinki-NLP/opus-mt-en-de')
            translated_text = translator(text_to_translate, max_length=40)[0]['translation_text']

            # Pokazanie przetłumaczonego tekstu
            st.success(f'{translated_text}')
        except Exception as e:
            st.error('Podczas tłumaczenia wystąpił błąd. Proszę spróbować jeszcze raz.')
            print(str(e))

elif option == 'Wydźwięk emocjonalny tekstu (eng)':
    text = st.text_area(label='Proszę podać tekst do przeanalizowania')
    if st.button('Przeanalizuj', type='primary') and text:
        try:
            classifier = pipeline("sentiment-analysis")
            answer = classifier(text)
            st.write(answer)
        except Exception as e:
            st.error('Podczas analizowania tekstu wystąpił błąd. Proszę spróbować jeszcze raz.')
            print(str(e))

st.subheader('Autor: Dominik Suszek')
st.subheader('Numer indeksu: s23396')