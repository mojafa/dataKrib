# **Sales Forecasting API with Prophet and FastAPI**

## **Project Overview**

This project is a time-series forecasting API that predicts daily sales for a specific store and department for the month of **November 2024**. It utilizes Facebook's **Prophet** library to model and forecast sales, incorporating additional regressors like holidays, temperature, fuel prices, CPI, unemployment rates, and promotional markdowns to enhance the accuracy of the predictions.

The API is built using **FastAPI**, providing endpoints to retrieve sales forecasts and insights into key factors impacting the predictions.

---

## **Table of Contents**

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Data Generation](#data-generation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [API Endpoints](#api-endpoints)
- [Example Response](#example-response)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Notes](#notes)
- [Acknowledgements](#acknowledgements)
- [Contact Information](#contact-information)

---

## **Features**

- **Time-Series Forecasting**: Predicts daily sales for November 2024 using Prophet.
- **Additional Regressors**: Incorporates holidays, temperature, fuel prices, CPI, unemployment rates, and promotional markdowns.
- **API Endpoints**:
  - `/predict`: Returns sales predictions with insights.
  - `/forecast_components`: Provides component plots of the forecast.
- **Synthetic Data Generation**: Generates synthetic training data for October 2024.
- **Insights**: Offers key factors impacting the forecasted sales with explanations.

---

## **Installation**

### **Prerequisites**

- **Python 3.7 or higher**
- **pip** package manager

### **Clone the Repository**

- Clone the repository from GitHub and navigate into the project directory.

### **Create a Virtual Environment (Recommended)**

- Create a virtual environment using `python3 -m venv venv`.
- Activate the virtual environment.
  - On macOS/Linux: `source venv/bin/activate`
  - On Windows: `venv\Scripts\activate`

### **Install Dependencies**

- Install the required Python packages listed in `requirements.txt` using `pip install -r requirements.txt`.

---

## **Data Generation**

Since we don't have actual sales data for the desired period, we'll generate synthetic data for **October 2024** to train the model.

### **Generate Synthetic Data**

- Run the data generation script `data_generation.py` to create `train.csv` and `features.csv` in the `data/` directory.
- Ensure that the `data/` directory exists in the project root.

### **Data Files**

- **`data/train.csv`**: Contains synthetic daily sales data for October 2024.
- **`data/features.csv`**: Contains synthetic feature data corresponding to the sales data.
- **`data/stores.csv`**: Contains store information.

---

## **Usage**

### **Running the Application**

- Start the FastAPI server using Uvicorn with the command `uvicorn main:app --reload`.
- The API will be available at `http://127.0.0.1:8000`.

### **API Endpoints**

#### **1. `/predict`**

- **Method**: GET
- **Description**: Returns sales predictions for November 2024 with insights on key factors impacting the forecast.
- **URL**: `http://127.0.0.1:8000/predict`

#### **2. `/forecast_components`**

- **Method**: GET
- **Description**: Provides a plot of the forecast components (trend, seasonality, etc.).
- **URL**: `http://127.0.0.1:8000/forecast_components`

---

## **Example Response**

**Request**:

- Send a GET request to `http://127.0.0.1:8000/predict`.

**Response**:

```json
{
  "sales_forecast": [
    {
      "date": "2024-11-01",
      "predicted_sales": 21045.67,
      "lower_bound": 19000.34,
      "upper_bound": 23000.99,
      "key_factors": [
        {
          "factor": "trend",
          "contribution": 20000.0,
          "percentage": 95.02,
          "explanation": "Overall trend in sales over time"
        },
        {
          "factor": "yearly",
          "contribution": 1045.67,
          "percentage": 4.98,
          "explanation": "Yearly seasonality"
        }
      ]
    },
    {
      "date": "2024-11-02",
      "predicted_sales": 21500.23,
      "lower_bound": 19500.12,
      "upper_bound": 23500.34,
      "key_factors": [
        {
          "factor": "trend",
          "contribution": 20000.0,
          "percentage": 93.02,
          "explanation": "Overall trend in sales over time"
        },
        {
          "factor": "yearly",
          "contribution": 1500.23,
          "percentage": 6.98,
          "explanation": "Yearly seasonality"
        }
      ]
    }
    // ... additional entries for each day in November 2024
  ]
}
```