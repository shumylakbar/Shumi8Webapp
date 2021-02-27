# import streamlit as st
# import pandas as pd
# from sklearn import datasets
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
#
# st.write("""
# # Simple C02 Emission Prediction App
# This app predicts the **Co2 Emission** type!
# """)
#
# st.sidebar.header('User Input Parameters')
#
# def user_input_features():
#     ENGINESIZE = st.sidebar.slider('ENGINESIZE', 4, 7.9, 5.4)
#     CYLINDERS = st.sidebar.slider('CYLINDERS', 2.0, 4.4, 3.4)
#     FUELCONSUMPTION_HWY = st.sidebar.slider('FUELCONSUMPTION_HWY', 1.0, 6.9, 1.3)
#     FUELCONSUMPTION_COMB = st.sidebar.slider('FUELCONSUMPTION_COMB', 0.1, 2.5, 0.2)
#     data = {'ENGINESIZE': ENGINESIZE,
#             'CYLINDERS': CYLINDERS,
#             'FUELCONSUMPTION_HWY': FUELCONSUMPTION_HWY,
#             'FUELCONSUMPTION_COMB': FUELCONSUMPTION_COMB}
#     features = pd.DataFrame(data, index=[0])
#     return features
#
# df = user_input_features()
#
# st.subheader('User Input parameters')
# st.write(df)
#
# cdf = pd.read_csv('C:/Users/Shumi-8/PycharmProjects/Streamlit-WebApp/FuelConsumption.csv')
# st.write(cdf)
# data = cdf[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
# predict = "CO2EMISSIONS"
# X = np.array(data.drop([predict], 1))
# Y = np.array(data[predict])
#
# clf = RandomForestClassifier()
# clf.fit(X, Y)
#
# prediction = clf.predict(df)
# prediction_proba = clf.predict_proba(df)
#
# st.subheader('Prediction of C02 Emission')
# st.write(prediction)
# #st.write(prediction)
#
# st.subheader('Prediction Probability')
# st.write(prediction_proba)
#






import streamlit as st
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import linear_model
import matplotlib
from matplotlib import style
from matplotlib import pyplot

st.write("""
# Students Grade Prediction App
This app predicts the **Students Grade**
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    traveltime = st.sidebar.slider('traveltime', 1, 4, 2)
    studytime = st.sidebar.slider('studytime', 1, 4, 2)
    failures = st.sidebar.slider('failures', 0, 3, 1)
    guardian = st.sidebar.slider('guardian', 0, 1, 0)
    data = {'traveltime': traveltime,
            'studytime': studytime,
            'failures': failures,
            'guardian': guardian}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

cdf = pd.read_csv("C:/Users/Shumi-8/PycharmProjects/Streamlit-WebApp/student-mat.csv", sep = ";")
cdf = cdf.replace('F', 0)
cdf = cdf.replace('M', 1)
cdf = cdf.replace('mother', 0)
cdf = cdf.replace('father', 1)
cdf = cdf.replace('other', .1)
st.write("**Student Dataset**", cdf)
data = cdf[['G3', 'traveltime', 'studytime', 'failures', 'guardian']]
predict = "G3"
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

reg = linear_model.LinearRegression()
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

reg.fit(X_train, Y_train)
acc = reg.score(X_test, Y_test)

predicted = reg.predict(df)
print(predicted)

st.subheader('Prediction of Students Grade')
st.write(predicted)
#st.write(prediction)

st.subheader('Prediction Accuracy')
st.write(acc)

style.use("ggplot")
pyplot.scatter(data["studytime"], data["G3"])
pyplot.xlabel("Study Time")
pyplot.ylabel("Final Grade")
st.pyplot()







