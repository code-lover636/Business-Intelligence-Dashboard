# 📊 Marketing Intelligence Dashboard

[Click to see hosted dashboard](https://dashboardmarket1.streamlit.app/)

This is an interactive BI dashboard built with **Streamlit** and **Plotly** to connect marketing activity (Facebook, Google, TikTok campaigns) with business outcomes (revenue, orders, profit).  

It is designed for **decision-makers** to understand where marketing spend works best, how it impacts customer acquisition, and how it contributes to overall business growth.

---

## 🚀 Features

- **Filters & Controls**  
  - Date range selector  
  - Channel selection (Facebook, Google, TikTok)  
  - Campaign, tactic, and state filters  

- **Headline KPIs**  
  - Total Revenue  
  - Marketing Spend  
  - Attributed Revenue  
  - ROAS (Return on Ad Spend)  
  - CAC (Customer Acquisition Cost)  

- **Visualizations & Insights**
  - Revenue vs Marketing Spend (trend over time)  
  - Marketing contribution to revenue  
  - Channel performance summary (CTR, CPC, CPM, ROAS)  
  - Campaign deep dive (top spenders & most efficient)  
  - Funnel view (Impressions → Clicks → Orders)  
  - Channel-specific trends (small multiples)  
  - State-level performance analysis  
  - Spend vs Attributed Revenue (scatter with regression)  

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) for interactive dashboard  
- [Plotly](https://plotly.com/python/) for charts  
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) for data wrangling  
- [scikit-learn](https://scikit-learn.org/) and [statsmodels](https://www.statsmodels.org/) for trendline regressions  

---

## 📂 Data Sources

The dashboard uses four datasets covering **120 days** of activity:

- `Facebook.csv`, `Google.csv`, `TikTok.csv` → Campaign-level marketing performance (impressions, clicks, spend, attributed revenue).  
- `business.csv` → Business performance metrics (orders, new orders, new customers, total revenue, gross profit, COGS).  

---

## ▶️ How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/your-repo/business-intelligence-dashboard.git
   cd business-intelligence-dashboard
    ```

2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run main.py
   ```

4. Open the local URL shown in the terminal

---

## 📌 Notes / Guidance

* Use the filters to explore time ranges, channels, tactics, states, or campaigns.
* **Key metrics:** CTR, CPC, CPM, ROAS, CAC, Marketing % of Revenue, Profit Margin.
* **Look for:**

  * Channels with high ROAS but high CAC → may be good to scale carefully.
  * Campaigns with low ROAS → candidates to pause.
  * Trends where CAC is rising → potential inefficiency risk.
* This app assumes daily-level merging between marketing and business via `date`.

  * For **campaign-level CAC attribution**, event-level or user-level joins would be required.
