# Greek Stock Tax Automation (IBKR)

This application automates the process of generating tax reports for the Greek Tax Authorities based on Interactive Brokers (IBKR) Activity Statements.

## Features
- **Streamlit Web UI**: Easy drag & drop interface.
- **Auto-ECB Rates**: Dynamically fetches the official European Central Bank (ECB) exchange rates for EUR conversions based on the dates present in the trade statement.
- **Moving Average Cost**: Calculates the exact rolling average cost basis for each stock position.
- **Capital Consumption (Πόθεν Έσχες)**: Automatically computes profit, sales revenue, and capital consumption.
- **Monthly 0.1% Tax Sheets**: Separately generates tabs (worksheets) grouping sales by month to assist with the monthly 1‰ (0.1%) sales tax declaration.
- **Greek Decimal Formats**: Safely generates pure numeric variables in the resulting Excel files, which open natively with decimal commas `,` in Greek locales.

## Local Installation

1. Make sure you have Python 3.9+ installed.
2. Clone this repository.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Cloud Deployment (Streamlit Community Cloud)

To deploy this app for free online so it can be accessed from any device without installing Python:
1. Create a GitHub account and upload these files (`app.py`, `requirements.txt`, `README.md`) to a new public or private repository.
2. Go to [Share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Click `New app`, select your repository, and set the Main file path to `app.py`.
5. Click **Deploy**. Your app will be live on a dedicated URL immediately.
