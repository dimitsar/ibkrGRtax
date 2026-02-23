import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import zipfile
import io as builtin_io

# -----------------
# ECB DOWNLOADER
# -----------------
def fetch_ecb_rates():
    """
    Downloads the official ECB time series ZIP, extracts the CSV, and loads it into a Pandas DataFrame.
    """
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with zipfile.ZipFile(builtin_io.BytesIO(response.content)) as z:
            # Usually the CSV is named 'eurofxref-hist.csv'
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df_ecb = pd.read_csv(f)
                
        df_ecb['Date'] = pd.to_datetime(df_ecb['Date']).dt.normalize()
        df_ecb = df_ecb.set_index('Date').sort_index()
        df_ecb = df_ecb.replace('N/A', np.nan)
        
        # Forward fill weekends/holidays based on the available date range
        full_idx = pd.date_range(start=df_ecb.index.min(), end=df_ecb.index.max())
        df_ecb = df_ecb.reindex(full_idx)
        df_ecb = df_ecb.ffill()
        
        if 'EUR' not in df_ecb.columns:
            df_ecb['EUR'] = 1.0
            
        return df_ecb
    except Exception as e:
        st.error(f"Σφάλμα κατά τη λήψη των ισοτιμιών ECB: {e}")
        return None

def get_ecb_rate(df_ecb, date, currency):
    date_norm = pd.to_datetime(date).normalize()
    if date_norm < df_ecb.index.min():
        date_norm = df_ecb.index.min()
    elif date_norm > df_ecb.index.max():
        date_norm = df_ecb.index.max()
        
    try:
        val = df_ecb.loc[date_norm, currency]
        if pd.isna(val):
            return 1.0
        return float(val)
    except KeyError:
        return 1.0

# -----------------
# IBKR PARSER
# -----------------
def load_ibkr_csv(uploaded_file):
    import csv
    trades = []
    dividends = []
    withholding_taxes = []
    
    # Needs to handle bytes from streamlit
    content = uploaded_file.getvalue().decode('utf-8').splitlines()
    reader = csv.reader(content)
    
    for parts in reader:
        if len(parts) < 2:
            continue
        if parts[0] == 'Trades' and parts[1] == 'Data' and parts[2] == 'Order':
            if parts[3] == 'Stocks':
                trades.append({
                    'Broker': 'IBKR',
                    'Νόμισμα': parts[4],
                    'Σύμβολο': parts[5],
                    'Ημερομηνία': pd.to_datetime(parts[6]),
                    'Quantity': float(parts[7].replace(',', '')),
                    'T. Price': float(parts[8].replace(',', '')),
                    'Proceeds': float(parts[10].replace(',', '')),
                    'Comm/Fee': float(parts[11].replace(',', ''))
                })
        elif parts[0] == 'Dividends' and parts[1] == 'Data':
            if not parts[2].startswith('Total'):
                dividends.append({
                    'Currency': parts[2],
                    'Date': pd.to_datetime(parts[3]),
                    'Description': parts[4],
                    'Amount': float(parts[5].replace(',', ''))
                })
        elif parts[0] == 'Withholding Tax' and parts[1] == 'Data':
            if not parts[2].startswith('Total'):
                withholding_taxes.append({
                    'Currency': parts[2],
                    'Date': pd.to_datetime(parts[3]),
                    'Description': parts[4],
                    'Amount': float(parts[5].replace(',', ''))
                })
                
    df_trades = pd.DataFrame(trades)
    df_divs = pd.DataFrame(dividends)
    df_withholding = pd.DataFrame(withholding_taxes)
    return df_trades, df_divs, df_withholding

# -----------------
# TAX LOGIC
# -----------------
def process_trades(df_trades, df_ecb):
    if df_trades.empty:
        return pd.DataFrame()
        
    df_trades = df_trades.sort_values(by=['Σύμβολο', 'Ημερομηνία']).reset_index(drop=True)
    results = []
    inventory = {}
    
    for idx, row in df_trades.iterrows():
        sym = row['Σύμβολο']
        if sym not in inventory:
            inventory[sym] = {'qty': 0.0, 'cost': 0.0}
            
        qty = row['Quantity']
        proceeds = row['Proceeds']
        comm = row['Comm/Fee']
        currency = row['Νόμισμα']
        date = row['Ημερομηνία']
        
        rate = get_ecb_rate(df_ecb, date, currency)
        
        oliki_axia_sym = proceeds + comm
        proceeds_eur = proceeds / rate if rate != 0 else 0
        comm_eur = comm / rate if rate != 0 else 0
        
        sale_tax_eur = 0.0
        if qty < 0 and proceeds_eur > 0:
            sale_tax_eur = -round(proceeds_eur * 0.001, 2)
            
        oliki_axia_eur = round(proceeds_eur + sale_tax_eur + comm_eur, 2)
        
        prev_qty = inventory[sym]['qty']
        prev_cost = inventory[sym]['cost']
        new_qty = prev_qty + qty
        
        if qty > 0:
            buy_value_eur = abs(oliki_axia_eur)
            inventory[sym]['cost'] += buy_value_eur
            inventory[sym]['qty'] = new_qty
        else:
            cost_per_share = prev_cost / prev_qty if prev_qty > 0 else 0
            sold_cost_eur = abs(qty) * cost_per_share
            inventory[sym]['cost'] -= sold_cost_eur
            inventory[sym]['qty'] = new_qty
            
        current_qty = inventory[sym]['qty']
        current_cost = inventory[sym]['cost']
        
        if current_qty > 0:
            avg_cost = current_cost / current_qty
        elif prev_qty > 0:
            avg_cost = prev_cost / prev_qty
        else:
            avg_cost = 0.0
        
        axia_agorasthenton = -oliki_axia_eur if oliki_axia_eur < 0 else 0.0
        kostos_polithenton = round(abs(qty) * (prev_cost/prev_qty if prev_qty>0 else 0), 2) if oliki_axia_eur > 0 else 0.0
        kostos_ypoloipon = round(current_cost, 2)
        kostos_polithenton_neg = -kostos_polithenton
        
        kerdos = max(oliki_axia_eur + kostos_polithenton_neg, 0.0) if oliki_axia_eur > 0 else 0.0
        esodo_polisis = (oliki_axia_eur - kerdos) if oliki_axia_eur > 0 else 0.0
        analosi = oliki_axia_eur if oliki_axia_eur < 0 else 0.0
        zimia = (esodo_polisis + kostos_polithenton_neg) if (esodo_polisis > 0 and esodo_polisis + kostos_polithenton_neg < 0) else 0.0
        
        res = {
            'Broker': row['Broker'],
            'Σύμβολο': sym,
            'Νόμισμα': currency,
            'Ημερομηνία': date,
            'Quantity': qty,
            'T. Price': row['T. Price'],
            'Proceeds': proceeds,
            'Comm/Fee': comm,
            'Ολική αξία συναλλαγής (Νόμισμα)': oliki_axia_sym,
            'Ισοτιμία EUR (ECB)': round(rate, 4),
            'Proceeds (EUR)': round(proceeds_eur, 2),
            'Φόρος Πώλησης 0,1% (EUR)': sale_tax_eur,
            'Comm/Fee (EUR)': round(comm_eur, 2),
            'Ολική Αξία Συναλλαγής (EUR)': oliki_axia_eur,
            'Υπόλοιπο μετοχών στο χαρτοφυλάκιο': current_qty,
            'Σταθμ.Μέσο Κόστος Υπολοίπου μετοχών': round(avg_cost, 4),
            'Αξία Αγορασθέντων μετοχών': round(axia_agorasthenton, 2),
            'Κόστος Πωληθέντων μετοχών': round(kostos_polithenton_neg, 2),
            'Κόστος Υπολοίπων μετοχών': kostos_ypoloipon,
            'Κέρδος από την Πώληση Ε1:659': round(kerdos, 2),
            'Έσοδο από Πώληση περιουσιακού στοιχείου Ε1:781': round(esodo_polisis, 2),
            'Ανάλωση Κεφαλαίου για αγορά περιουσιακού στοιχείου Ε1:743': round(analosi, 2),
            'Ζημία Πωλήσεων': round(zimia, 2)
        }
        results.append(res)
        
    return pd.DataFrame(results)


def process_dividends(df_divs, df_with, df_ecb):
    div_results = []
    if not df_divs.empty:
        for idx, row in df_divs.iterrows():
            date = row['Date']
            curr = row['Currency']
            amount = row['Amount']
            desc = row['Description']
            
            w_tax = 0.0
            if not df_with.empty:
                matches = df_with[(df_with['Date'] == date) & (df_with['Currency'] == curr)]
                if not matches.empty:
                    w_tax = matches['Amount'].sum()
            
            rate = get_ecb_rate(df_ecb, date, curr)
            amount_eur = round(amount / rate, 2) if rate else 0
            w_tax_eur = round(w_tax / rate, 2) if rate else 0
            
            div_results.append({
                'Broker': 'IBKR',
                'Ημερομηνία': date,
                'Νόμισμα': curr,
                'Περιγραφή': desc,
                'Μεικτό μέρισμα (ξένο νόμισμα)': amount,
                'Παρακρατημένος φόρος (ξένο νόμισμα)': w_tax,
                'Ισοτιμία (EUR)': round(rate, 4),
                'Αξία μερίσματος (EUR)': amount_eur,
                'Αξία παρακρατημένου φόρου (EUR)': w_tax_eur
            })
    return pd.DataFrame(div_results)


# -----------------
# STREAMLIT APP
# -----------------
st.set_page_config(page_title="Greek Stock Tax Automation", page_icon="📈", layout="wide")

st.title("📈 Αυτοματοποίηση Υπολογισμών Φορολογίας (IBKR)")
st.markdown("Ανεβάστε το Activity Statement CSV αρχείο από την Interactive Brokers. Η εφαρμογή θα κατεβάσει **αυτόματα** τις ισοτιμίες από την Ευρωπαϊκή Κεντρική Τράπεζα (ΕΚΤ) και θα εξάγει το Excel για το Ε1 και τους φόρους 0.1% ανά μήνα.")

uploaded_file = st.file_uploader("Σύρετε και αφήστε (Drag & Drop) το αρχείο CSV του IBKR εδώ", type=['csv'])

if uploaded_file is not None:
    with st.spinner("Λήψη Ισοτιμιών από Ευρωπαϊκή Κεντρική Τράπεζα (ΕΚΤ)..."):
        df_ecb = fetch_ecb_rates()
        
    if df_ecb is not None:
        with st.spinner("Επεξεργασία δεδομένων και δημιουργία Excel..."):
            df_trades, df_divs, df_with = load_ibkr_csv(uploaded_file)
            res_trades = process_trades(df_trades, df_ecb)
            res_divs = process_dividends(df_divs, df_with, df_ecb)
            
            # Create Excel in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # 1. Γενικό Φύλλο Μετοχών
                if not res_trades.empty:
                    res_trades.to_excel(writer, sheet_name='Μετοχές_Ετήσιο', index=False)
                    
                    # 2. Μηνιαία Φύλλα 0.1% Φόρου (Μόνο Πωλήσεις)
                    sales_df = res_trades[res_trades['Quantity'] < 0].copy()
                    if not sales_df.empty:
                        # Add YearMonth column for grouping
                        sales_df['YearMonth'] = pd.to_datetime(sales_df['Ημερομηνία']).dt.to_period('M')
                        months = sales_df['YearMonth'].unique()
                        
                        for month in sorted(months):
                            month_df = sales_df[sales_df['YearMonth'] == month].copy()
                            month_df = month_df.drop(columns=['YearMonth'])
                            
                            sheet_name = f'Πωλήσεις_{month.strftime("%Y_%m")}'
                            month_df.to_excel(writer, sheet_name=sheet_name, index=False)
                            
                            # Write sum of tax at the bottom
                            worksheet = writer.sheets[sheet_name]
                            last_row = len(month_df) + 1
                            tax_col_idx = month_df.columns.get_loc('Φόρος Πώλησης 0,1% (EUR)') + 1
                            tax_sum = month_df['Φόρος Πώλησης 0,1% (EUR)'].sum()
                            
                            # Add Total Row
                            worksheet.cell(row=last_row + 2, column=tax_col_idx - 1, value="ΣΥΝΟΛΟ ΦΟΡΟΥ ΜΗΝΑ:")
                            worksheet.cell(row=last_row + 2, column=tax_col_idx, value=tax_sum)

                # 3. Μερίσματα
                if not res_divs.empty:
                    res_divs.to_excel(writer, sheet_name='Μερίσματα', index=False)
            
            processed_data = output.getvalue()
            
            st.success("✅ Η επεξεργασία ολοκληρώθηκε επιτυχώς!")
            st.balloons()
            
            # Provide Download Button
            st.download_button(
                label="📥 Κατέβασμα Αρχείου Excel (Tax_Report.xlsx)",
                data=processed_data,
                file_name="Tax_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            with st.expander("Προεπισκόπηση Δεδομένων (Πρώτες 10 εγγραφές)"):
                st.dataframe(res_trades.head(10))
