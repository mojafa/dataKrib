from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from prophet import Prophet
import holidays
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse

app = FastAPI()

# Load and prepare the data
def load_data():
    # Load datasets
    train_df = pd.read_csv('data/train.csv')
    features_df = pd.read_csv('data/features.csv')
    stores_df = pd.read_csv('data/stores.csv')

    # Convert Date columns
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    features_df['Date'] = pd.to_datetime(features_df['Date'])

    # Ensure 'IsHoliday' is of the same data type
    train_df['IsHoliday'] = train_df['IsHoliday'].astype(bool)
    features_df['IsHoliday'] = features_df['IsHoliday'].astype(bool)

    # Merge datasets
    df = pd.merge(train_df, features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = pd.merge(df, stores_df, on='Store', how='left')

    # Rename Date column and Daily_Sales
    df = df.rename(columns={'Date': 'ds', 'Daily_Sales': 'y'})

    # Filter data for a specific Store and Department
    store = 1
    dept = 1
    df = df[(df['Store'] == store) & (df['Dept'] == dept)]

    # Select relevant columns including MarkDowns
    markdown_columns = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df = df[['ds', 'y', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] + markdown_columns]

    # Convert IsHoliday to integer
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # Handle missing values: Fill NaNs in MarkDowns with 0
    df[markdown_columns] = df[markdown_columns].fillna(0)

    # Drop rows with other missing values
    df = df.dropna()

    return df

# Train the model
def train_model():
    df = load_data()
    model = Prophet(daily_seasonality=True)

    # Add regressors
    model.add_regressor('IsHoliday')
    model.add_regressor('Temperature')
    model.add_regressor('Fuel_Price')
    model.add_regressor('CPI')
    model.add_regressor('Unemployment')

    # Add markdowns as regressors
    markdown_columns = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_columns:
        model.add_regressor(col)

    model.fit(df)
    return model, df

# Initialize the model at startup
model, df = train_model()

# Define explanations for components
component_explanations = {
    'trend': 'Overall trend in sales over time',
    'seasonal': 'Combined seasonal effects',
    'weekly': 'Weekly seasonality',
    'yearly': 'Yearly seasonality',
    'holidays': 'Impact of holidays on sales',
    'extra_regressors_IsHoliday': 'Holiday indicator',
    'extra_regressors_Temperature': 'Effect of temperature',
    'extra_regressors_Fuel_Price': 'Effect of fuel price',
    'extra_regressors_CPI': 'Consumer Price Index impact',
    'extra_regressors_Unemployment': 'Unemployment rate impact',
    'extra_regressors_MarkDown1': 'Promotional markdown 1',
    'extra_regressors_MarkDown2': 'Promotional markdown 2',
    'extra_regressors_MarkDown3': 'Promotional markdown 3',
    'extra_regressors_MarkDown4': 'Promotional markdown 4',
    'extra_regressors_MarkDown5': 'Promotional markdown 5'
}

@app.get('/predict')
def predict():
    try:
        # Create future dataframe for dates from Nov 1, 2024, to Nov 30, 2024
        future_dates = pd.date_range(start='2024-11-01', end='2024-11-30', freq='D')
        future = pd.DataFrame({'ds': future_dates})

        # Prepare future regressors
        future_regressors = {
            'IsHoliday': [],
            'Temperature': [],
            'Fuel_Price': [],
            'CPI': [],
            'Unemployment': []
        }

        # Include markdowns
        markdown_columns = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        for col in markdown_columns:
            future_regressors[col] = []

        us_holidays = holidays.UnitedStates()

        for ds in future['ds']:
            # Determine if the date is a holiday
            is_holiday = int(ds in us_holidays)
            future_regressors['IsHoliday'].append(is_holiday)
            # Use plausible values or last known values
            future_regressors['Temperature'].append(np.random.uniform(40, 60))
            future_regressors['Fuel_Price'].append(np.random.uniform(2.5, 3.5))
            future_regressors['CPI'].append(np.random.uniform(200, 250))
            future_regressors['Unemployment'].append(np.random.uniform(5, 10))
            # For markdowns, assume zero or planned values
            for col in markdown_columns:
                future_regressors[col].append(0)  # Assuming no promotions planned

        # Add regressors to future dataframe
        for regressor in future_regressors:
            future[regressor] = future_regressors[regressor]

        # Make predictions
        forecast = model.predict(future)

        # Sort the forecast by date
        forecast.sort_values(by='ds', inplace=True)

        # Extract component contributions
        components = ['trend', 'seasonal', 'weekly', 'yearly', 'holidays']
        for regressor in future_regressors.keys():
            components.append(f'extra_regressors_{regressor}')

        # Prepare insights with threshold
        insights = []
        for index, row in forecast.iterrows():
            # Collect component contributions
            component_contributions = {comp: row.get(comp, 0) for comp in components}

            # Calculate threshold (e.g., 10% of yhat)
            threshold = 0.10 * abs(row['yhat']) if row['yhat'] != 0 else 0

            # Identify significant factors
            significant_factors = []
            for k, v in component_contributions.items():
                if abs(v) >= threshold:
                    explanation = component_explanations.get(k, 'No explanation available')
                    significant_factors.append({
                        'factor': k,
                        'contribution': round(v, 2),
                        'percentage': round((v / row['yhat']) * 100, 2) if row['yhat'] != 0 else 0,
                        'explanation': explanation
                    })

            # Sort factors by absolute contribution
            significant_factors.sort(key=lambda x: abs(x['contribution']), reverse=True)

            insights.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'predicted_sales': round(row['yhat'], 2),
                'lower_bound': round(row['yhat_lower'], 2),
                'upper_bound': round(row['yhat_upper'], 2),
                'key_factors': significant_factors
            })

        # Plot components
        plot_components(forecast)
        return {'sales_forecast': insights}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def plot_components(forecast):
    fig = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('forecast_components.png')
    plt.close(fig)

@app.get('/forecast_components')
def get_forecast_components():
    try:
        return FileResponse('forecast_components.png', media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=404, detail="Component plot not found")
