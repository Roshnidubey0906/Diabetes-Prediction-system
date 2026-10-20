
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------------
# 1. PAGE SETTINGS
# ----------------------------------
st.set_page_config(
    page_title="Diabetes AI Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------------
# 2. CUSTOM DARK NAVY + TEAL DESIGN
# ----------------------------------
st.markdown("""
<style>
.stApp {
    background: #0B1220;
    color: #F1F5F9;
}
[data-testid="stSidebar"] {
    background: #111C30;
    border-right: 1px solid #263B52;
}
h1, h2, h3 {
    color: #55E6D0 !important;
}
p, label {
    color: #D2DFEC;
}
[data-testid="stMetric"] {
    background: #14263A;
    border: 1px solid #29435B;
    border-radius: 14px;
    padding: 18px;
}
[data-testid="stForm"] {
    background: #111C30;
    border: 1px solid #29435B;
    border-radius: 16px;
    padding: 22px;
}
div.stButton > button,
div.stFormSubmitButton > button {
    background: #19B8A6;
    color: #071522;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #55E6D0;
    color: #071522;
    border: none;
}
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #172940;
}
[data-testid="stDataFrame"] {
    border: 1px solid #29435B;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# 3. LOAD YOUR EXISTING MODEL + SCALER
# ----------------------------------
# Keep both .pkl files in the same folder as app.py.
model = joblib.load("diabetes_svm_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# ----------------------------------
# 4. CREATE SESSION HISTORY
# ----------------------------------
# History stays during the current app session.
if "history" not in st.session_state:
    st.session_state.history = []

# Store the latest prediction separately.
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

# ----------------------------------
# 5. SIDEBAR NAVIGATION
# ----------------------------------
st.sidebar.markdown("# 🩺 Diabetes AI")
st.sidebar.caption("SMART HEALTH DASHBOARD")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "New Prediction", "History"]
)

st.sidebar.divider()
st.sidebar.markdown("### Model Information")
st.sidebar.write("**Algorithm:** SVM")
st.sidebar.write("**Kernel:** Linear")
st.sidebar.write("**Interface:** Streamlit")
st.sidebar.caption("Educational project — not a medical diagnosis.")

# ----------------------------------
# 6. DASHBOARD PAGE
# ----------------------------------
if page == "Dashboard":

    st.title("Your Health Dashboard")
    st.write(
        "A simple machine-learning interface for "
        "exploring diabetes prediction."
    )

    # These are your previously measured test metrics.
    col1, col2, col3 = st.columns(3)

    col1.metric("Test Accuracy", "77%")
    col2.metric("Weighted F1-Score", "0.76")
    col3.metric("Diabetic Recall", "52%")

    st.caption(
        "Metrics are from your previous test evaluation; "
        "they are not recalculated in this app."
    )

    st.divider()
    st.subheader("Model Performance")

    left, right = st.columns([1.1, 1])

    # Confusion matrix from your previous test results.
    with left:
        st.markdown("#### Confusion Matrix")

        cm = np.array([
            [91, 9],
            [26, 28]
        ])

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#14263A")
        ax.set_facecolor("#14263A")

        ax.imshow(cm, cmap="GnBu")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(
            ["Non-Diabetic", "Diabetic"],
            color="white"
        )
        ax.set_yticklabels(
            ["Non-Diabetic", "Diabetic"],
            color="white"
        )

        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual", color="white")
        ax.set_title("Test Results", color="#55E6D0")

        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row, str(cm[row, col]),
                    ha="center", va="center",
                    color="black", fontsize=14
                )

        ax.tick_params(colors="white")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### What do these numbers mean?")

        st.info(
            "**91** — Non-diabetic cases correctly predicted."
        )
        st.info(
            "**28** — Diabetic cases correctly predicted."
        )
        st.warning(
            "**26** — Diabetic cases missed by the model."
        )
        st.write(
            "The model needs improvement at identifying "
            "diabetic cases. Do not use it for medical decisions."
        )

    st.divider()

    # Show the most recent result, if available.
    st.subheader("Latest Prediction")

    if st.session_state.latest_prediction is None:
        st.write(
            "No prediction yet. Open New Prediction "
            "to try the model."
        )
    else:
        result = st.session_state.latest_prediction

        if result["Prediction"] == "Diabetic":
            st.warning("Latest model result: Diabetic")
        else:
            st.success("Latest model result: Non-Diabetic")

# ----------------------------------
# 7. NEW PREDICTION PAGE
# ----------------------------------
elif page == "New Prediction":

    st.title("New Prediction")
    st.write(
        "Enter the eight features in the form below."
    )

    with st.form("diabetes_prediction_form"):

        st.subheader("🧑 Patient Information")

        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0, max_value=20, value=1
            )

            glucose = st.number_input(
                "Glucose Level",
                min_value=0.0, value=120.0
            )

            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=0.0, value=70.0
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0.0, value=20.0
            )

        with col2:
            insulin = st.number_input(
                "Insulin",
                min_value=0.0, value=79.0
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0, value=25.0
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0, value=0.5
            )

            age = st.number_input(
                "Age",
                min_value=1, max_value=120, value=30
            )

        submitted = st.form_submit_button(
            " Predict Result",
            use_container_width=True
        )

    # Run the model only when the form is submitted.
    if submitted:

        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]])

        # Use the same saved scaler as the trained model.
        scaled_data = scaler.transform(input_data)

        # Get the model prediction.
        prediction = int(model.predict(scaled_data)[0])

        result = (
            "Diabetic" if prediction == 1
            else "Non-Diabetic"
        )

        # Save prediction in current session history.
        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Age": int(age),
            "Glucose": float(glucose),
            "BMI": float(bmi),
            "Prediction": result
        }

        st.session_state.history.append(record)
        st.session_state.latest_prediction = record

    # Show the most recent prediction on this page.
    if st.session_state.latest_prediction is not None:
        result = st.session_state.latest_prediction["Prediction"]

        st.divider()
        st.subheader("Prediction Result")

        if result == "Diabetic":
            st.warning("Model Prediction: Diabetic")
        else:
            st.success("Model Prediction: Non-Diabetic")

        st.caption(
            "This is an educational model result, "
            "not a medical diagnosis."
        )


# 8. HISTORY PAGE

elif page == "History":

    st.title("Prediction History")
    st.write(
        "Results generated during the current app session."
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        col1, col2 = st.columns(2)
        col1.metric(
            "Total Predictions",
            len(history_df)
        )
        col2.metric(
            "Diabetic Predictions",
            int((history_df["Prediction"] == "Diabetic").sum())
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # Let the user download the history.
        csv_data = history_df.to_csv(index=False)

        st.download_button(
            "⬇ Download History CSV",
            data=csv_data,
            file_name="diabetes_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.latest_prediction = None
            st.rerun()

    else:
        st.info(
            "No history yet. Make a prediction first."
        )
