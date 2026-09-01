import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. Model Training Engine
# ---------------------------------------------------------
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

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------------------------------------------------
# 2. Evaluation Logic Function
# ---------------------------------------------------------
def evaluate_progress(commits, active_hrs, reopen_rate, response_time, overtime):
    input_features = pd.DataFrame(
        [[commits, active_hrs, reopen_rate, response_time, overtime]],
        columns=['daily_commits', 'active_hours', 'ticket_reopen_rate', 
                 'avg_response_time_hrs', 'overtime_hours']
    )
    
    prediction = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]
    risk_score = probabilities[1] * 100
    
    if prediction == 1:
        status = "⚠️ HIGH RISK OF DELAY"
        recommendation = "Recommendations: Check for developer burnout, reduce open ticket rework, or reassign high-complexity tasks."
    else:
        status = "✅ PROJECT ON TRACK"
        recommendation = "Recommendations: Steady velocity detected. Maintain current task allocation."
        
    return status, f"{risk_score:.1f}%", recommendation

# ---------------------------------------------------------
# 3. Gradio Interface Construction
# ---------------------------------------------------------
demo = gr.Interface(
    fn=evaluate_progress,
    inputs=[
        gr.Slider(minimum=0, maximum=15, value=4, step=1, label="Daily Commits"),
        gr.Slider(minimum=1.0, maximum=12.0, value=6.5, step=0.5, label="Active Work Hours"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.10, step=0.05, label="Ticket Reopen Rate (0.0 to 1.0)"),
        gr.Slider(minimum=0.5, maximum=24.0, value=2.0, step=0.5, label="Avg Response Time (Hours)"),
        gr.Slider(minimum=0, maximum=8, value=1, step=1, label="Overtime Hours per Day")
    ],
    outputs=[
        gr.Textbox(label="Project Risk Status"),
        gr.Textbox(label="Calculated Delay Probability"),
        gr.Textbox(label="Actionable Evaluation Advice")
    ],
    title="Smart Project Progress Monitoring System",
    description="Evaluate project health based on behavioral telemetric indicators."
)

# Launch for local or cloud hosting
if __name__ == "__main__":
    demo.launch()
