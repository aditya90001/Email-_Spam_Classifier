import streamlit as st
import joblib
import numpy as np

# Load model + vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize counters in session state
if "spam_count" not in st.session_state:
    st.session_state.spam_count = 0
if "ham_count" not in st.session_state:
    st.session_state.ham_count = 0

# Apply dark theme styling
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    textarea, .stButton>button {
        background-color: #1e1e1e;
        color: #ffffff;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #333333;
    }
    .stProgress>div>div>div>div {
        background-color: #00ff00;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title("📩 SMS Spam Detector")
    st.markdown("### 🤖 Detect whether your SMS is **Spam** or **Not Spam** in real time!")

    # Input area
    sms = st.text_area("✍️ Enter your SMS message below:", height=150)

    if st.button("🔍 Predict"):
        if sms.strip() != "":
            sms_vector = vectorizer.transform([sms])
            prediction = model.predict(sms_vector)[0]

            # Probability
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(sms_vector)[0]
                spam_proba = proba[1]
                ham_proba = proba[0]
            else:
                spam_proba = 1.0 if prediction == 1 else 0.0
                ham_proba = 1.0 - spam_proba

            # Result
            if prediction == 1:
                st.markdown("🚨 **This message is SPAM!**")
                st.session_state.spam_count += 1
            else:
                st.markdown("✅ **This message is NOT SPAM!**")
                st.session_state.ham_count += 1

            # Probability bar and values
            st.progress(int(spam_proba * 100))
            st.write(f"📊 **Spam Probability:** {spam_proba:.2f} | **Not Spam Probability:** {ham_proba:.2f}")

            # Stats
            st.write(f"📈 Total Predictions → Spam: {st.session_state.spam_count} | Not Spam: {st.session_state.ham_count}")
        else:
            st.warning("⚠️ Please enter a message before predicting!")

if __name__ == "__main__":
    main()
