import pandas as pd
from joblib import Parallel, delayed
from binance.client import Client
from datetime import datetime, UTC
import time
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

symbol = 'BTCUSDT'
interval = '5m'
horizont = 180

list_of_n_changepoints = [10, 90, 100, 110, 120]
list_of_n_changepoint_prior_scale = [0.001, 0.03, 0.04, 0.05, 0.1]
list_of_seasonality_mode = ['additive', 'multiplicative']
list_of_daily_fourier_order = [3, 5]
list_of_hourly_fourier_order = [1, 2, 3]
list_of_seasonality_prior_scale = [5.0, 10.0, 20]


client = Client()
df_result = pd.DataFrame(columns=['n_changepoints', 'changepoint_prior_scale', 'seasonality_mode', 'daily_fourier_order', 'hourly_fourier_order', 'seasonality_prior_scale', 'MAE'])

def get_binance_data(symbol='BTCUSDT', interval='5m', lookback = '12 days'):
    now = datetime.now(UTC)
    past = now - pd.to_timedelta(lookback)

    df = pd.DataFrame(client.get_historical_klines(symbol, interval, str(past), str(now)
    ), columns=['open_time','Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol','is_best_match'])
    
    df['Date']= pd.to_datetime(df['close_time'], unit='ms')
    return df[['Date','Open', 'High', 'Low', 'Close', 'Volume']]  


def safe_get_binance_data(symbol='BTCUSDT', interval='5m', lookback = '12 days', max_retries=10, base_delay=5):
    retries = 0
    while True:
        try:
            return get_binance_data(interval=interval, lookback=lookback)
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                raise RuntimeError(f"Max retries reached. Last error: {e}")
            delay = base_delay * (2 ** (retries - 1))  # Exponential backoff
            print(f"Error: {e}. Retrying in {delay} sec...")
            time.sleep(delay)



data = safe_get_binance_data(symbol=symbol, interval='5m', lookback = '20 days')

df_train = data[['Date', 'Close']].dropna(subset=['Close']).copy()
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train = df_train.astype({'Close': 'float'})
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})
q_low = df_train['y'].quantile(0.005)
q_high = df_train['y'].quantile(0.995)
df_train = df_train[(df_train['y'] >= q_low) & (df_train['y'] <= q_high)]

df_test = df_train.iloc[-3000:-horizont].copy()
df_train = df_train.iloc[-3000:].copy()

def train_and_evaluate(params, df_train, df_test, horizont):
    """Function to be parallelized"""
    model_param = {
        "n_changepoints": params['n_points'],
        "daily_seasonality": True,
        "weekly_seasonality": False,
        "yearly_seasonality": False,
        "seasonality_mode": params['seasonality_mode'],
        "changepoint_prior_scale": params['scale'],
        "seasonality_prior_scale": params['seasonal_prior_scale'],
        "holidays_prior_scale": 10.0,
        "growth": "logistic"
    }
    
    m = Prophet(**model_param)
    df_test['cap'] = df_test["y"].max() + df_test["y"].std() * 0.05
    m.add_country_holidays(country_name='US')
    m.add_seasonality(name='daily', period=1, fourier_order=params['daily_fourier_order'])
    m.add_seasonality(name='weekly', period=7, fourier_order=3)
    m.add_seasonality(name='hourly', period=1/24, fourier_order=params['hourly_fourier_order'])
    m.fit(df_test)
    future = m.make_future_dataframe(periods=horizont, freq='5min')
    future['cap'] = df_test['cap'].max()
    forecast = m.predict(future)
    true_y = df_train['y'].iloc[-horizont:]
    predict_y = forecast['yhat'].iloc[-horizont:]
    mae = mean_absolute_error(true_y, predict_y)
    
    return params['n_points'], params['scale'], params['seasonality_mode'], params['seasonal_prior_scale'], params['daily_fourier_order'], params['hourly_fourier_order'], mae

# Prepare parameter grid
param_grid = [
    {
        'n_points': i,
        'scale': j,
        'seasonality_mode': k,
        'daily_fourier_order': l,
        'hourly_fourier_order': m,
        'seasonal_prior_scale': n
    } 
             for i in list_of_n_changepoints 
             for j in list_of_n_changepoint_prior_scale
             for k in list_of_seasonality_mode
             for l in list_of_daily_fourier_order
             for m in list_of_hourly_fourier_order
             for n in list_of_seasonality_prior_scale
]

# Run in parallel (adjust n_jobs to your CPU cores)
#6-core CPU (no hyper-threading) → 6 logical cores → Set n_jobs=5.
results = Parallel(n_jobs=5, verbose=10)( 
    delayed(train_and_evaluate)(params, df_train, df_test, horizont)
    for params in param_grid
)

# Convert results to DataFrame
df_result = pd.DataFrame(results, columns=['n_changepoints', 'changepoint_prior_scale', 'seasonality_mode', 'daily_fourier_order', 'hourly_fourier_order', 'seasonality_prior_scale', 'MAE'])     
df_result = df_result.sort_values(by='MAE', ascending=True).copy()


df_result.to_csv(
    '/home/lex/Documents/Jupyter/get_df_binance_with_forecast/grid_search_hyperparameters_for_Prophet.csv',     # Full file path
    index=True,                     # Save index column
    sep=',',                        # Use comma as delimiter (default)
    float_format='%.4f',            # Format floating point numbers
    encoding='utf-8'                # Set encoding
)

min_mae_row = df_result.loc[df_result['MAE'].idxmin()]
print(min_mae_row)

