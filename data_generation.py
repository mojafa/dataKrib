import pandas as pd
import numpy as np
import holidays

def generate_features_csv():
    # Date range for features (October 1, 2024 to October 31, 2024)
    dates = pd.date_range(start='2024-10-01', end='2024-10-31', freq='D')
    
    # Initialize lists to hold data
    data = []
    
    us_holidays = holidays.UnitedStates()
    
    for date in dates:
        is_holiday = date in us_holidays
        record = {
            'Store': 1,
            'Date': date.strftime('%Y-%m-%d'),
            'Temperature': np.random.uniform(50, 70),
            'Fuel_Price': np.random.uniform(2.5, 3.5),
            'MarkDown1': np.random.uniform(0, 5000),
            'MarkDown2': np.random.uniform(0, 5000),
            'MarkDown3': np.random.uniform(0, 5000),
            'MarkDown4': np.random.uniform(0, 5000),
            'MarkDown5': np.random.uniform(0, 5000),
            'CPI': np.random.uniform(200, 250),
            'Unemployment': np.random.uniform(5, 10),
            'IsHoliday': is_holiday
        }
        data.append(record)
    
    features_df = pd.DataFrame(data)
    features_df.to_csv('data/features.csv', index=False)
    print("features.csv generated successfully.")

def generate_train_csv():
    # Date range for sales data (October 1, 2024 to October 31, 2024)
    dates = pd.date_range(start='2024-10-01', end='2024-10-31', freq='D')
    
    # Initialize lists to hold data
    sales_data = []
    
    us_holidays = holidays.UnitedStates()
    
    for date in dates:
        is_holiday = date in us_holidays
        base_sales = 20000 + np.random.uniform(-5000, 5000)
        if is_holiday:
            base_sales *= 1.2  # Increase by 20% on holidays
        sales_record = {
            'Store': 1,
            'Dept': 1,
            'Date': date.strftime('%Y-%m-%d'),
            'Daily_Sales': base_sales,
            'IsHoliday': is_holiday
        }
        sales_data.append(sales_record)
    
    train_df = pd.DataFrame(sales_data)
    train_df.to_csv('data/train.csv', index=False)
    print("train.csv generated successfully.")

def generate_stores_csv():
    # Generate store data
    stores_data = [
        {
            'Store': 1,
            'Type': 'A',
            'Size': 151315
        }
    ]
    
    stores_df = pd.DataFrame(stores_data)
    stores_df.to_csv('data/stores.csv', index=False)
    print("stores.csv generated successfully.")
# Generate the data
generate_features_csv()
generate_train_csv()
generate_stores_csv()

