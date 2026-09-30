# MacroScope

An economic dashboard for CPI, unemployment, the federal funds rate, real GDP, industrial production, and consumer sentiment. It includes levels, normalized indexes, changes, correlations, lead/lag exploration, source metadata, caching, and synthetic fallback.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Public mode uses the Federal Reserve Bank of St. Louis FRED CSV service. Always consult each series' official notes and release/revision details.

