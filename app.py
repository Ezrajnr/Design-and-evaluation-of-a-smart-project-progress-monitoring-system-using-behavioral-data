import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. Page Configuration & Gradio-Like CSS Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Project Progress Monitoring System",
    layout="wide"
)

# Inject custom CSS to replicate the light-gray boxed UI and orange slider styling
st.markdown("""
    <style>
    /* Main container title alignment */
    .main-title {
        text-align: center;
        font-weight: 700;
        font-size: 2rem;
        color: #1F2937;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: left;
        color: #4B5563;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
    /* Panel Containers */
    div[data-testid="stVerticalBlock"] > div.element-container {
        width: 100%;
    }
    .output-box {
        background-color: #FAFAFA;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .output-label {
        font-size: 0.85rem;
        color: #6B7280;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .output-content {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 6px;
        padding: 10px;
        min-height: 42px;
        font-size: 0.95rem;
        color: #111827;
    }
    /* Custom Slider Accent Color */
    div[data-baseweb="slider"] > div {
        color: #FF5500 !important;
    }
    /* Action Buttons Custom Styling */
    div.stButton > button[kind="primary"] {
        background-color: #FF5500;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #E5E7EB;
        color: #374151;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Model Training Engine
# ---------------------------------------------------------
@st.cache_resource
def train_model():
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

    df['delay_risk'] = np.where(
        (df['daily_commits'] < 2) | (df['ticket_reopen_rate'] > 0.25) | (df['overtime_hours'] > 3), 1, 0
    )

    X = df[['daily_commits', 'active_hours', 'ticket_reopen_rate', 'avg_response_time_hrs', 'overtime_hours']]
    y = df['delay_risk']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

model = train_model()

# Initialize session state for outputs and defaults
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ---------------------------------------------------------
# 3. Header
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>Smart Project Progress Monitoring System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Evaluate project health based on behavioral telemetric indicators.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. Two-Column Layout (Inputs Left | Outputs Right)
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    with st.container():
        st.markdown('<div style="background-color: #FAFAFA; border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px;">', unsafe_allow_html=True)
        
        commits = st.slider("Daily Commits", min_value=0, max_value=15, value=4, step=1)
        active_hrs = st.slider("Active Work Hours", min_value=1.0, max_value=12.0, value=6.5, step=0.5)
        reopen_rate = st.slider("Ticket Reopen Rate (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.10, step=0.05)
        response_time = st.slider("Avg Response Time (Hours)", min_value=0.5, max_value=24.0, value=2.0, step=0.5)
        overtime = st.slider("Overtime Hours per Day", min_value=0, max_value=8, value=1, step=1)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            clear_btn = st.button("Clear", use_container_width=True, type="secondary")
        with btn_col2:
            submit_btn = st.button("Submit", use_container_width=True, type="primary")

with col_right:
    # Logic for Clear / Submit actions
    if clear_btn:
        st.session_state.submitted = False
    elif submit_btn:
        st.session_state.submitted = True

    if st.session_state.submitted:
        input_data = pd.DataFrame(
            [[commits, active_hrs, reopen_rate, response_time, overtime]],
            columns=['daily_commits', 'active_hours', 'ticket_reopen_rate', 'avg_response_time_hrs', 'overtime_hours']
        )
        prediction = model.predict(input_data)[0]
        risk_score = model.predict_proba(input_data)[0][1] * 100

        if prediction == 1:
            status_val = "⚠️ HIGH RISK OF DELAY"
            advice_val = "Recommendations: Check for developer burnout, reduce open ticket rework, or reassign high-complexity tasks."
        else:
            status_val = "✅ PROJECT ON TRACK"
            advice_val = "Recommendations: Steady velocity detected. Maintain current task allocation."
        
        prob_val = f"{risk_score:.1f}%"
    else:
        status_val = ""
        prob_val = ""
        advice_val = ""

    # Boxed Output Panel
    st.markdown(f"""
        <div style="background-color: #FAFAFA; border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px;">
            <div class="output-label">Project Risk Status</div>
            <div class="output-content">{status_val}</div>
            <br>
            <div class="output-label">Calculated Delay Probability</div>
            <div class="output-content">{prob_val}</div>
            <br>
            <div class="output-label">Actionable Evaluation Advice</div>
            <div class="output-content">{advice_val}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    flag_btn = st.button("Flag", use_container_width=True, type="secondary")

# ---------------------------------------------------------
# 5. Footer Signature
# ---------------------------------------------------------
st.markdown("""
    <div style="text-align: center; color: #9CA3AF; font-size: 0.8rem; margin-top: 30px;">
        Runs ↺ &nbsp;•&nbsp; Built with Streamlit &nbsp;•&nbsp; Settings ⚙
    </div>
""", unsafe_allow_html=True)
