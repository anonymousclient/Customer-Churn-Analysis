# 📊 Customer Churn Analysis System

A **Streamlit-based web application** that analyses customer churn patterns from uploaded datasets and presents an interactive dashboard with KPIs, charts, insights, and recommendations.

> Built as a BCA Minor Project.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 File Upload | Upload CSV or Excel files via the sidebar |
| 🧹 Auto Cleaning | Missing values filled, columns renamed, types fixed |
| 🔄 Transformation | Age, Balance, and Credit Score grouped automatically |
| 📈 KPI Cards | Total customers, churned, retained, and churn rate |
| 📊 Interactive Charts | 6 Plotly charts (bar + pie) update with filters |
| 🔍 Sidebar Filters | Filter by Gender, Country, Activity Status |
| 💡 Auto Insights | Key patterns highlighted in plain English |
| 🚀 Recommendations | Actionable suggestions to reduce churn |
| 🎨 Dark Theme | Professional dark UI with custom styling |

---

## 🗂️ Project Structure

```
Minorrrrrr/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── sample_dataset.csv      # Sample data for testing
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Theme & server settings
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Steps

1. **Open a terminal** in the project folder.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. The app will open in your browser at **http://localhost:8501**.

---

## 📋 Expected Dataset Format

The uploaded CSV/Excel file should contain these columns:

| Column | Type | Description |
|---|---|---|
| `customer_id` | int | Unique customer identifier |
| `credit_score` | int | Credit score (300–850) |
| `country` | str | Customer's country |
| `gender` | str | Male / Female |
| `age` | int | Customer's age |
| `tenure` | int | Years with the bank |
| `balance` | float | Account balance |
| `products_number` | int | Number of products used |
| `credit_card` | int | Has credit card (0/1) |
| `active_member` | int | Is active (0/1) |
| `churn` | int | Churned (0/1) |

> A ready-to-use `sample_dataset.csv` is included in the project.

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Streamlit** — Web UI framework
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Plotly** — Interactive visualisations
- **Matplotlib / Seaborn** — Additional charting

---

## 📸 How to Use

1. Launch the app with `streamlit run app.py`.
2. Click **Browse files** in the sidebar and upload `sample_dataset.csv`.
3. View the raw data preview and the cleaned/transformed data.
4. Explore the KPI cards and interactive charts.
5. Use sidebar filters to drill down into specific segments.
6. Read the auto-generated insights and recommendations.

---

## 📄 License

This project is created for academic purposes (BCA Minor Project).
