import streamlit as st
import pandas as pd
import joblib

# Set browser tab title
st.set_page_config(page_title="Airline Satisfaction Predictor", page_icon="✈️", layout="wide")

# 1. Load the pre-trained machine learning model
@st.cache_resource
def load_model():
    return joblib.load("lgbm_best_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file 'lgbm_best_model.pkl' not found. Ensure it is saved in your project folder.")
    st.stop()

# 2. Header and Description
st.title("✈️ Airline Passenger Satisfaction Predictor")
st.write("Adjust the flight details and full survey ratings below to predict passenger satisfaction in real-time.")

st.divider()

# 3. User Inputs (Sidebar & Main Form)
st.sidebar.header("📋 Passenger & Flight Profile")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
type_of_travel = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.sidebar.selectbox("Class", ["Business", "Eco", "Eco Plus"])
age = st.sidebar.slider("Passenger Age", 7, 85, 35)
flight_dist = st.sidebar.slider("Flight Distance (miles)", 100, 4000, 1200)

st.sidebar.header("⏱️ Delays")
dep_delay = st.sidebar.number_input("Departure Delay (minutes)", min_value=0, max_value=1500, value=0)
arr_delay = st.sidebar.number_input("Arrival Delay (minutes)", min_value=0, max_value=1500, value=0)

st.subheader("⭐ Full Inflight & Service Ratings (1 = Poor, 5 = Excellent)")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Convenience & Digital**")
    wifi = st.slider("Inflight Wi-Fi Service", 0, 5, 3)
    dep_arr_time = st.slider("Departure/Arrival Time Convenient", 0, 5, 3)
    booking = st.slider("Ease of Online Booking", 0, 5, 3)
    gate = st.slider("Gate Location", 0, 5, 3)

with col2:
    st.markdown("**Comfort & Inflight**")
    food = st.slider("Food and Drink", 0, 5, 3)
    boarding = st.slider("Online Boarding", 0, 5, 3)
    seat = st.slider("Seat Comfort", 0, 5, 3)
    entertainment = st.slider("Inflight Entertainment", 0, 5, 3)

with col3:
    st.markdown("**Crew & Support**")
    onboard = st.slider("On-board Service", 0, 5, 3)
    legroom = st.slider("Leg Room Service", 0, 5, 3)
    baggage = st.slider("Baggage Handling", 0, 5, 3)
    checkin = st.slider("Check-in Service", 0, 5, 3)
    inflight_service = st.slider("Inflight Service", 0, 5, 3)
    cleanliness = st.slider("Cleanliness", 0, 5, 3)

# 4. Predict Button
if st.button("🔮 Predict Satisfaction", type="primary", use_container_width=True):
    # Assemble dataframe containing the exact 21 features in the exact training order
    input_data = pd.DataFrame([{
        "Age": age,
        "Flight Distance": flight_dist,
        "Inflight wifi service": wifi,
        "Departure/Arrival time convenient": dep_arr_time,
        "Ease of Online booking": booking,
        "Gate location": gate,
        "Food and drink": food,
        "Online boarding": boarding,
        "Seat comfort": seat,
        "Inflight entertainment": entertainment,
        "On-board service": onboard,
        "Leg room service": legroom,
        "Baggage handling": baggage,
        "Checkin service": checkin,  # Exact name match without the hyphen
        "Cleanliness": cleanliness,
        "Departure Delay in Minutes": dep_delay,
        "Arrival Delay in Minutes": arr_delay,
        "Gender": gender,
        "Customer Type": customer_type,
        "Type of Travel": type_of_travel,
        "Class": travel_class
    }])

    # Cast text categorical columns to 'category' dtype
    categorical_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
    for col in categorical_cols:
        input_data[col] = input_data[col].astype("category")

    # Generate prediction from LightGBM booster
    satisfaction_prob = model.predict(input_data)[0]

    st.divider()
    
    # Display results
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(label="Satisfaction Probability", value=f"{satisfaction_prob * 100:.1f}%")
    with res_col2:
        if satisfaction_prob >= 0.5:
            st.success("🎉 **Passenger is predicted to be Satisfied!**")
        else:
            st.error("⚠️ **Passenger is predicted to be Neutral / Dissatisfied.**")
