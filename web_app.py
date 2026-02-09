import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression

# --- 1. THE BRAIN (Setup & Training) ---
@st.cache_data
def load_and_train():
    # 1. Fetch the data from the internet
    dataset = fetch_ucirepo(id=320)
    
    # 2. Get the "Clues" (features) and the "Answers" (targets)
    features = dataset.data.features
    targets = dataset.data.targets
    
    # 3. Combine them into one big table (DataFrame)
    full_df = pd.concat([features, targets], axis=1)
    
    # 4. Now we pick the columns we want from that big table
    clues = ['G1', 'G2', 'absences', 'studytime', 'failures']
    X = full_df[clues]
    y = full_df['G3']
    
    # 5. Train the brain
    model = LinearRegression()
    model.fit(X, y)
    return model
# Call the function to build the model brain
model = load_and_train()
# --- 2. THE INTERFACE (Sidebar) ---
st.title("🎓 Student Success Predictor")

st.sidebar.header("Student Input")
g1 = st.sidebar.slider("Grade 1 (0-20)", 0, 20, 10)
g2 = st.sidebar.slider("Grade 2 (0-20)", 0, 20, 10)
absences = st.sidebar.slider("Absences", 0, 50, 5)
study_time = st.sidebar.selectbox("Weekly Study Time (1-4)", [1, 2, 3, 4])
failures = st.sidebar.number_input("Past Failures", 0, 3, 0)

# --- 3. THE PREDICTION ---
# Create a tiny table with the counselor's input
input_data = pd.DataFrame([[g1, g2, absences, study_time, failures]], 
                          columns=['G1', 'G2', 'absences', 'studytime', 'failures'])

prediction = model.predict(input_data)[0]

# Display the result
# Create a nice visual metric
st.metric(label="Predicted Performance", value=f"{prediction:.1f} / 20")
st.header(f"Predicted Final Grade: {prediction:.1f} / 20")

# --- 4. COUNSELING INSIGHTS ---
if prediction < 10:
    st.error("⚠️ This student is at risk of failing.")
    
    # Check for Absences 🚩
    if absences > 10:
        st.info("💡 Insight: High absences are pulling the grade down. Consider an attendance intervention.")
    
    # Check for Study Time 📖
    if study_time == 1:
        st.info("💡 Insight: Low study time detected. The student may benefit from a structured study plan.")
        
    # Check for Past Failures 🏗️
    if failures > 0:
        st.info("💡 Insight: Student has a history of past failures. They may need foundational support or tutoring.")
else:
    st.success("✅ This student is on track to pass!")