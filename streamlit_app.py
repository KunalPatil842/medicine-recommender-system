import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load model
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].iloc[0]
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()[0]
    med = medications[medications['Disease'] == dis]['Medication'].values.tolist()
    die = diets[diets['Disease'] == dis]['Diet'].values.tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].values.tolist()
    return desc, pre, med, die, wrkout

# Model prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Main Streamlit app
def main():
    st.title('Disease Prediction App')

    symptoms = st.text_input('Enter symptoms separated by commas (e.g., itching, chills):')

    if st.button('Predict'):
        if symptoms == "":
            st.error("Please enter symptoms")
        else:
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            st.success(f'Predicted disease: {predicted_disease}')
            st.write(f'Description: {dis_des}')
            st.write('Precautions:')
            for precaution in precautions:
                st.write(f'- {precaution}')
            st.write('Medications:')
            for medication in medications:
                st.write(f'- {medication}')
            st.write('Recommended Diet:')
            for diet in rec_diet:
                st.write(f'- {diet}')
            st.write('Workout:')
            st.write(workout)

if __name__ == '__main__':
    main()
