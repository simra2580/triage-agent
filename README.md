🧠 Multi-Domain Support Triage Agent

An intelligent system that automatically analyzes and triages customer support tickets across multiple domains — HackerRank, Claude, and Visa — using rule-based risk detection and corpus-driven retrieval.

---

🚀 Features

- 📥 Accepts support tickets from a CSV file
- ⚠️ Detects risk signals (fraud, account issues, urgency)
- 🔍 Retrieves relevant information from a support corpus
- 🧠 Classifies request type
- 🔁 Decides whether to:
  - Auto-reply
  - Escalate to human support
- 📊 Outputs structured results in CSV format

---

🏗️ System Architecture

Input (CSV)
   ↓
Risk Classifier (rule-based)
   ↓
Corpus Retriever (TF-IDF)
   ↓
Decision Engine
   ↓
Output (results.csv)

---

🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn (TF-IDF retrieval)
- Rule-based NLP

---

▶️ How to Run

1. Install dependencies

pip install -r requirements.txt

2. Run the agent

python main.py

---

📂 Input Format

The input CSV must contain:

issue,subject,company
Payment failed,,Visa
Cannot login,,Claude

---

📊 Output Format

The system generates:

- "status" → replied / escalated
- "product_area"
- "response"
- "justification"
- "request_type"

Saved to:

output/results.csv

---

🧠 Design Decisions

- Uses rule-based classification for deterministic and reliable evaluation
- Uses TF-IDF retrieval to ground responses in support documentation
- Avoids external API dependencies for reproducibility and stability

---

👩‍💻 Author

Simra Begum