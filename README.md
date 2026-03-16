# Supply Chain Intelligence Hub: End-to-End Predictive Analytics 🚚

## 1. Project Overview
This project is an integrated intelligence suite designed to optimize e-commerce operations using the **Olist Brazilian E-Commerce dataset**. It features two specialized AI engines that translate raw transaction data into actionable supply chain strategies.

## 2. Module 1: Demand Forecasting (The Revenue Engine)
**Goal:** Forecast weekly product demand to optimize inventory and prevent stockouts.

* **The Model:** A **Hybrid Stacking Ensemble** combining **XGBoost Regressor** with a **Ridge Regression** meta-learner.
* **Key Innovation:** Engineered a **'Local Delivery Ratio'** feature to account for regional logistics efficiency and its impact on customer purchasing behavior.
* **Target:** Weekly units sold by category and state.

## 3. Module 2: Desirability Predictor (The Risk Engine)
**Goal:** Identify high-risk orders likely to result in low satisfaction or returns before they happen.

* **The Model:** **XGBoost Classifier** integrated with **NLP (Natural Language Processing)**.
* **Technical Highlight:** Uses **TfidfVectorizer** to analyze customer comments and an **'Optimal Price Score'** to determine if price-to-shipping ratios are driving dissatisfaction.
* **Addressing Imbalance:** Utilized `scale_pos_weight` to handle the class imbalance typical in customer return/review data.

## 4. Key Business Metrics
- **Inventory Precision:** High-fidelity weekly forecasting to reduce holding costs.
- **Sentiment-to-Action:** Converts text-based feedback into a numerical "Desirability Score" (0-100%).
- **Logistics Insights:** Quantifies the "Shipping Pain Score" to identify states with systematic delivery issues.

## 5. Technology Stack
- **Languages:** Python (Pandas, NumPy)
- **Machine Learning:** Scikit-Learn, XGBoost
- **NLP:** NLTK (Portuguese stopword filtering)
- **Deployment:** Streamlit (Multi-page Dashboard Architecture)

## 6. Setup & Usage
1. **Download Data:** Place the Olist dataset CSVs in the `/data` folder.
2. **Train Models:** ```bash
   python scripts/train_demand_model.py
   python scripts/train_desirability_model.py
