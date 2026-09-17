# AutoIQ: Used Car Market Analytics & Price Prediction

AutoIQ is a machine learning-powered dashboard for analysing used-car market data and predicting vehicle selling prices. The application combines Business Intelligence techniques, interactive visualisations, and predictive analytics to support data-driven decision-making.

## Features

- Interactive Streamlit dashboard
- Used-car market analysis
- Vehicle price prediction
- Dynamic filtering and exploration
- Interactive Plotly visualisations
- Machine learning model comparison
- Automated testing with PyTest

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib
- PyTest

## Dataset

The project uses the CarDekho Used Car Dataset containing information on vehicle listings, including:

- Manufacturing year
- Fuel type
- Transmission
- Ownership history
- Mileage
- Engine specifications
- Selling price

Dataset Source:
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

## Machine Learning Models

The following regression models were evaluated:

- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression

Models were assessed using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Gradient Boosting delivered the strongest overall performance and was selected as the final model.

## Project Structure

```text
app.py
data_processing.py
model.py
visualizations.py
insights.py
train_model.py
requirements.txt
autoiq_price_model.joblib
Car_details_v3.csv
tests/
README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

The application will launch at:

```text
http://localhost:8501
```

## Run Tests

```bash
pytest
```

## Limitations

- Results depend on dataset quality and coverage.
- Market conditions may change over time.
- Historical listing data may contain regional or sampling biases.

## Future Enhancements

- Real-time market data integration
- Cloud deployment
- Advanced ensemble models
- Geographical price analysis

