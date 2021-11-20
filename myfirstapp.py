import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier

st.write("""
# Iris Flower Prediction App
Allow us to predict the Iris flower type.
  Please insert values for the Sepal Length, Sepal Width, Petal Length and Petal Width. :)
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.number_input('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.number_input('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.number_input('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.number_input('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


st.subheader('User Input parameters')
st.dataframe(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = AdaBoostClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.dataframe(iris.target_names)

st.subheader('Prediction')
st.dataframe(iris.target_names[prediction])
#st.dataframe(prediction)

st.subheader('Prediction Probability')
st.dataframe(prediction_proba)