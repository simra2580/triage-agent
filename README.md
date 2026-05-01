🧠 Multi-Domain Support Triage Agent

This project is an AI-based system that automatically analyzes and classifies customer support tickets across different domains (HackerRank, Claude, Visa).

---

🚀 What it does

- Takes support issues from a CSV file
- Detects risk level (fraud, account issues, etc.)
- Classifies the request type
- Decides whether to:
  - Reply automatically
  - Escalate to human support
- Generates structured output

---

🏗️ Architecture

Input (CSV)
   ↓
Risk Classifier (rule-based)
   ↓
Retriever (support corpus)
   ↓
Decision Engine
   ↓
Output (results.csv)

---

▶️ How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Run the project:

python main.py

---

📂 Input Format

CSV file should contain:

issue,subject,company
Payment failed,,Visa
Cannot login,,Claude

---

📊 Output

The system generates:

- status (replied / escalated)
- product_area
- response
- justification
- request_type

Saved in:

output/results.csv


---

👩‍💻 Author

Simra Begum