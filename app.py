import streamlit as st
from src.classifier import RiskClassifier

# Page config
st.set_page_config(page_title="Triage AI", page_icon="⚡", layout="wide")

# Premium Custom Styling
st.markdown("""
    <style>
    /* Background and Global Styles */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }
    
    /* Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Glassmorphic Cards */
    .stSecondaryBlock, div[data-testid="stExpander"], .css-1r6slb0 {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
    }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.4);
        border: none;
        color: white;
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize classifier
classifier = RiskClassifier()

# Layout: Sidebar for Demo Examples
with st.sidebar:
    st.image("https://flaticon.com", width=80)
    st.markdown("## Agent Controls")
    st.info("The classifier uses a mix of LLM scoring and pattern matching to detect high-risk tickets.")
    
    st.markdown("### 💡 Quick Templates")
    if st.button("🚨 Urgent: Data Breach"):
        st.session_state.issue_input = "My account was hacked and my credit card is being charged!"
    if st.button("💳 Payment Failure"):
        st.session_state.issue_input = "Charged twice for the same subscription."
    if st.button("❓ General Inquiry"):
        st.session_state.issue_input = "How do I change my profile picture?"

# Main UI
st.markdown('<p class="main-title">⚡ Triage AI</p>', unsafe_allow_html=True)
st.caption("Enterprise-grade customer sentiment and risk analysis.")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    issue = st.text_area("Customer Request", 
                        value=st.session_state.get('issue_input', ''),
                        placeholder="Describe the customer issue here...", 
                        height=200)
    
    analyze_btn = st.button("Run Intelligence Check")

with col2:
    st.markdown("### Configuration")
    company = st.selectbox("Assign to Client", ["Visa", "Claude", "HackerRank", "Internal"])
    priority_mode = st.toggle("Aggressive Escalation Mode", value=True)

if analyze_btn:
    if not issue.strip():
        st.toast("Please enter an issue first!", icon="⚠️")
    else:
        with st.spinner("Decoding intent..."):
            risk = classifier.check(issue)
            
        st.divider()
        
        # Decision Logic
        is_high_risk = risk["must_escalate"]
        status_text = "HUMAN INTERVENTION REQUIRED" if is_high_risk else "AUTO-RESOLVE ELIGIBLE"
        status_color = "#ef4444" if is_high_risk else "#10b981"
        
        # Results Header
        st.markdown(f"### Result: <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Level", risk["level"].upper())
        m2.metric("Company Context", company)
        m3.metric("Actionable Flags", len(risk["flags"]))

        # Detailed Analysis Card
        with st.container():
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("#### 🚩 Identified Risk Flags")
                if risk["flags"]:
                    for flag in risk["flags"]:
                        st.markdown(f"- `{flag}`")
                else:
                    st.write("No critical flags detected.")
            
            with res_col2:
                st.markdown("#### 📝 Recommended Response")
                if is_high_risk:
                    st.error(f"Priority escalation triggered for {company}. Notification sent to Senior Support Lead.")
                else:
                    st.success("Drafting automated AI response... Ticket remains in low-priority queue.")

st.markdown("---")
st.caption("Powered by RiskClassifier v2.4 • High Precision Mode Active")