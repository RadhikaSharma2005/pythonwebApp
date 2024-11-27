import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./diabetes.csv")

# Updated Header with Modern Styling
st.markdown(
    """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .app-title {
            text-align: center;
            font-size: 2.5rem;
            color: #ffffff;
            background-color: #1f77b4;
            padding: 15px;
            border-radius: 10px;
            animation: fadeIn 2s ease-in-out;
            margin-bottom: 20px;
        }

        .app-subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #333333;
            margin-bottom: 15px;
        }

        .card {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
        }

        .sidebar-title {
            font-size: 1.1rem;
            color: #1f77b4;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .separator {
            height: 2px;
            background: linear-gradient(to right, #1f77b4, #ff7f0e);
            margin: 20px 0;
        }

        .progress-bar {
            background-color: #1f77b4;
        }

        .result-text {
            text-align: center;
            font-size: 1.8rem;
            font-weight: bold;
            color: #ffffff;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        }

        .not-diabetic {
            background-color: #28a745;
        }

        .diabetic {
            background-color: #dc3545;
        }
    </style>
    <div class="app-title">Diabetes Prediction App</div>
    <div class="app-subtitle">
        Predict if a patient is diabetic based on their health data.
    </div>
    <div class="separator"></div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Patient Data Input")
st.sidebar.markdown("<div class='sidebar-title'>Enter the following details:</div>", unsafe_allow_html=True)


def calc():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    bmi = st.sidebar.number_input('BMI', min_value=0, max_value=67, value=20)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=79)
    age = st.sidebar.number_input('Age', min_value=21, max_value=88, value=33)

    output = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    return pd.DataFrame(output, index=[0])


user_data = calc()

st.markdown("<div class='card'><strong>Patient Data Summary:</strong></div>", unsafe_allow_html=True)
st.write(user_data)

x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

progress = st.progress(0)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
progress.progress(100)

result = rf.predict(user_data)

st.markdown("<div class='card'><strong>Prediction Result:</strong></div>", unsafe_allow_html=True)
output = "You are not Diabetic" if result[0] == 0 else "You are Diabetic"
result_class = "not-diabetic" if result[0] == 0 else "diabetic"
st.markdown(f"<div class='result-text {result_class}'>{output}</div>", unsafe_allow_html=True)

accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.markdown("<div class='card'><strong>Model Accuracy:</strong></div>", unsafe_allow_html=True)
st.write(f"{accuracy:.2f}%")
