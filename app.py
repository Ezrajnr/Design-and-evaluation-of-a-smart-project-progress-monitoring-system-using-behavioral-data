import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Project Progress Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Smart Project Progress Monitoring System")
st.markdown("Evaluate project delay risk and project health based on team behavioral telemetry.")

# ---------------------------------------------------------
# 2. Model Training Engine (Cached)
# ---------------------------------------------------------
@st.cache_resource
def train_behavioral_model():
    # Synthetic Behavioral Telemetry Generation
    np.random.seed(42)
    num_records = 600

    data = {
        'daily_commits': np.random.poisson(lam=4, size=num_records),
        'active_hours': np.round(np.random.normal(loc=6.5, scale=1.5, size=num_records), 1),
        'ticket_reopen_rate': np.round(np.random.uniform(0.0, 0.4, size=num_records), 2),
        'avg_response_time_hrs': np.round(np.random.exponential(scale=2.0, size=num_records), 1),
        'overtime_hours': np.random.choice([0, 1, 2, 3, 4], size=num_records, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    }
    df = pd.DataFrame(data)

    # Risk Label Generation Logic (1 = Delay Risk, 0 = Safe)
    df['delay_risk'] = np.where(
        (df['daily_commits'] < 2) | (df['ticket_reopen_rate'] > 0.25) | (df['overtime_hours'] > 3), 1, 0
    )

    X = df[['daily_commits', 'active_hours', 'ticket_reopen_rate', 'avg_response_time_hrs', 'overtime_hours']]
    y = df['delay_risk']

    # Train Classification Engine
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    return clf, X.columns.tolist()

model, feature_names = train_behavioral_model()

# ---------------------------------------------------------
# 3. Sidebar - Telemetry Input Parameters
# ---------------------------------------------------------
st.sidebar.header("🕹️ Team Behavioral Metrics")

commits = st.sidebar.slider(
    "Daily Commits", 
    min_value=0, max_value=15, value=4, step=1,
    help="Average number of code commits per developer daily."
)

active_hrs = st.sidebar.slider(
    "Active Work Hours", 
    min_value=1.0, max_value=12.0, value=6.5, step=0.5,
    help="Logged active working hours per day."
)

reopen_rate = st.sidebar.slider(
    "Ticket Reopen Rate", 
    min_value=0.0, max_value=1.0, value=0.10, step=0.05,
    help="Ratio of completed tasks that were reopened due to bugs or missing requirements."
)

response_time = st.sidebar.slider(
    "Avg Response Time (Hours)", 
    min_value=0.5, max_value=24.0, value=2.0, step=0.5,
    help="Average time taken to respond to code reviews and communications."
)

overtime = st.sidebar.slider(
    "Overtime Hours per Day", 
    min_value=0, max_value=8, value=1, step=1,
    help="Extra working hours beyond standard schedule."
)

# ---------------------------------------------------------
# 4. Prediction Engine & Layout
# ---------------------------------------------------------
input_df = pd.DataFrame(
    [[commits, active_hrs, reopen_rate, response_time, overtime]],
    columns=feature_names
)

prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]
risk_score = probabilities[1] * 100

st.subheader("📋 Evaluation Dashboard")

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.error("⚠️ **Status: HIGH RISK OF DELAY**")
    else:
        st.success("✅ **Status: PROJECT ON TRACK**")

with col2:
    st.metric(label="Calculated Delay Probability", value=f"{risk_score:.1f}%")

st.markdown("---")

# ---------------------------------------------------------
# 5. Recommendation Engine
# ---------------------------------------------------------
st.subheader("💡 Behavioral Insights & Advice")

if prediction == 1:
    st.warning(
        "**Action Recommended:** High delay risk detected. Key risk factors include low commit rates, "
        "excessive ticket reopens, or developer overtime overload.\n\n"
        "* **Strategy:** Review ticket scope clarity, reduce work-in-progress limits, and rebalance workload to avoid burnout."
    )
else:
    st.info(
        "**Optimal Performance:** Current behavioral telemetry indicates steady velocity and balanced work patterns.\n\n"
        "* **Strategy:** Maintain standard sprint pacing and current task distribution."
    )
