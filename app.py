import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# Mendapatkan API Key dari Tiingo
api_key = '60b72460f66eed7ec99a062003fe6a87bb8e94bb'

def get_historical_data(ticker, start_date, end_date):
    url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

def main():
    st.title('Analisis Tren Harga Saham')

    ticker = st.text_input('Masukkan ticker saham (misal: AAPL untuk Apple):')
    start_date = st.date_input('Pilih tanggal awal:')
    end_date = st.date_input('Pilih tanggal akhir:')

    if st.button('Proses'):
        historical_data = get_historical_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        df['timestamp'] = df.index.map(datetime.timestamp)
        X = df['timestamp'].values.reshape(-1, 1)
        y = df['close'].values

        model = LinearRegression()
        model.fit(X, y)
        df['trend'] = model.predict(X)

        # Menghitung metrik evaluasi
        mae = mean_absolute_error(y, df['trend'])
        mse = mean_squared_error(y, df['trend'])
        r2 = r2_score(y, df['trend'])

        # Menampilkan plot menggunakan Matplotlib
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['close'], label='Harga Penutupan')
        plt.plot(df.index, df['trend'], label='Tren', linestyle='--')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga Penutupan (USD)')
        plt.title(f'Analisis Tren Harga Saham {ticker}')
        plt.legend()
        plt.grid(True)

        st.pyplot(plt)  # Menampilkan plot di Streamlit

        st.write(f'MAE: {mae:.2f}')
        st.write(f'MSE: {mse:.2f}')
        st.write(f'R^2 Score: {r2:.2f}')

if __name__ == '__main__':
    main()
