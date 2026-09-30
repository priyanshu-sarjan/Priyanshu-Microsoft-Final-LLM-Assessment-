# Employee Sentiment Analysis & Retention Risk Modeling

## 📌 Project Overview
This repository delivers an end-to-end machine learning and natural language processing (NLP) workflow for analyzing workplace communications, scoring employee sentiment over time, ranking employee engagement, identifying retention flight risks, and building predictive regression models to forecast monthly sentiment scores.

---

## 🛠️ Environment Setup & Installation

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/priyanshu-sarjan/Priyanshu-Microsoft-Final-LLM-Assessment-.git
cd employee-sentiment-analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Jupyter Notebook
```bash
jupyter notebook notebooks/Employee_Sentiment_Analysis.ipynb
```

---

## 🔬 Methodology & Architecture

1. **Sentiment Labeling (NLP Task 1)**: Combined NLTK VADER compound sentiment analysis with domain-specific profanity/tone detection. Scores map to:
   - **Positive (+1)**: Compound score >= 0.05
   - **Neutral (0)**: -0.05 < compound score < 0.05
   - **Negative (-1)**: Compound score <= -0.05 or presence of strong negative workplace markers.

2. **Exploratory Data Analysis (EDA Task 2)**: Standardized dates using `pd.to_datetime(df['date'], errors='coerce')`, handled null values, extracted `year_month` periods, and plotted label distributions and monthly trends.

3. **Monthly Sentiment Scoring (Task 3)**: Monthly scores reset at the beginning of each calendar month for each employee using `df.groupby(['from', 'year_month'])['sentiment_score'].sum()`.

4. **Employee Ranking (Task 4)**: Monthly rankings produced for Top 3 Positive and Top 3 Negative employees, sorted first by score in descending order and then alphabetically by email.

5. **Flight Risk Identification (Task 5)**: Identified employees sending >= 4 negative messages within any 30-day rolling window (`rolling('30D', on='date')`).

6. **Predictive Modeling (Task 6)**: Engineered 5 monthly aggregated features (`total_messages`, `avg_character_length`, `avg_word_count`, `negative_ratio`, `positive_ratio`) and trained a `LinearRegression` model to predict `monthly_sentiment_score`.

---

## 📊 Key Findings & Summary Tables

### 1. Overall Sentiment Distribution
- **Positive**: 71 messages (60.7%)
- **Neutral**: 32 messages (27.4%)
- **Negative**: 14 messages (12.0%)

### 2. Top Monthly Employee Rankings (Sample Representative Periods)

| Period | Top 3 Positive Employees (Email / Score) | Top 3 Negative Employees (Email / Score) |
|---|---|---|
| **2000-06** | 1. `john.arnold@enron.com` (+2)<br>2. `kayne.coulter@enron.com` (+1)<br>3. `sally.beck@enron.com` (+1) | 1. `rhonda.denton@enron.com` (-1)<br>2. `kayne.coulter@enron.com` (+1)<br>3. `sally.beck@enron.com` (+1) |
| **2000-08** | 1. `bobette.riner@ipgdirect.com` (+2)<br>2. `john.arnold@enron.com` (+1)<br>3. `johnny.palmer@enron.com` (+1) | 1. `don.baughman@enron.com` (0)<br>2. `john.arnold@enron.com` (+1)<br>3. `johnny.palmer@enron.com` (+1) |
| **2001-01** | 1. `bobette.riner@ipgdirect.com` (+2)<br>2. `sally.beck@enron.com` (+2)<br>3. `lydia.delgado@enron.com` (+1) | 1. `patti.thompson@enron.com` (0)<br>2. `lydia.delgado@enron.com` (+1)<br>3. `rhonda.denton@enron.com` (+1) |

### 3. Flight Risk Findings (Task 5)
- **Criterion**: >= 4 negative emails in any 30-day rolling window.
- **Outcome**: **0 employees flagged as active flight risks** in the 117-message sample dataset. The maximum negative message frequency observed was 1-2 negative emails within any 30-day window (e.g. Don Baughman and Sally Beck).

---

## 📈 Model Performance Results (Task 6)

- **R^2 Score**: 0.8949 (Explains 89.49% of variance in monthly sentiment scores)
- **RMSE**: 0.3099
- **MAE**: 0.2516

### Feature Impact Coefficients:
- **`positive_ratio`**: `+1.2214` (Strongest positive driver of monthly score)
- **`negative_ratio`**: `-1.0023` (Strongest negative driver of monthly score)
- **`total_messages`**: `+0.5732` (Higher activity correlates with positive engagement)
- **`avg_character_length`**: `+0.0027`
- **`avg_word_count`**: `-0.0129`

---

## 📁 Repository Directory Structure

```
employee-sentiment-analysis/
├── data/
│   ├── test.csv
│   └── test_labeled.csv
├── notebooks/
│   └── Employee_Sentiment_Analysis.ipynb
├── visualizations/
│   ├── sentiment_distribution.png
│   ├── monthly_sentiment_trend.png
│   ├── top_negative_employees.png
│   ├── top_positive_employees.png
│   └── linear_regression_fit.png
├── final_report/
│   └── Final_Employee_Sentiment_Report.docx
├── .env.example
├── requirements.txt
└── README.md
```

---
*Author: Priyanshu Sarjan*  
*Evaluation Submission for Microsoft / Glynac AI Assessment*
