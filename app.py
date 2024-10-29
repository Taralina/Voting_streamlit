import streamlit as st
import joblib
import numpy as np

# Загрузка пайплайна
pipeline = joblib.load('catboost_pipeline.pkl')

# Заголовок приложения
st.title('CatBoost Classifier')

# Ввод данных от пользователя
features = st.text_input("Введите характеристики (через запятую):")

if st.button('Предсказать'):
    if features:
        feature_list = [float(x) for x in features.split(',')]
        prediction = pipeline.predict([feature_list])
        st.write(f'Предсказание: {prediction[0]}')
    else:
        st.write("Пожалуйста, введите данные.")
