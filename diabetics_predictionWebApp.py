# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:44:51 2024

@author: dell precision
"""


import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('D:/AI Classification Projects/Diabetes Prediction Model/trained_model.sav', 'rb'))


def diabetics_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
      return "The person is not Diabetic"
    else:
      return "The person is Diabetic"

  
    
  
def main():
    st.title("Diabetics Prediction Web App")
         
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Level")
    SkinThickness = st.text_input("SkinThickness Level")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Level")
    Age = st.text_input("Age of the Person")
    
    
    diagnosis = ''
    
    if st.button("Diabetics Prediction Result"):
        diagnosis = diabetics_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age ])
        
    
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    