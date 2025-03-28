# Delhi Power Demand Forecasting

This project implements a deep learning solution for forecasting power demand in Delhi using LSTM (Long Short-Term Memory) neural networks. The system analyzes historical power demand data with 5-minute intervals and provides predictions for future demand through an interactive web interface.

## Features

- LSTM-based time series forecasting with 5-minute granularity
- Interactive Streamlit web interface for real-time predictions
- Historical data visualization and analysis
- Weather data integration for improved accuracy
- Customizable forecast horizon (1-24 hours)
- Detailed performance metrics and statistics

## Data

The project can work with two types of data:
1. **Full Dataset** (not included in repository due to size):
   - File: `powerdemand_5min_2021_to_2024_with weather.csv`
   - Contains complete historical power demand data
   - Contact repository owner for access to full dataset

2. **Sample Dataset** (included for demonstration):
   - File: `sample_power_demand.csv`
   - Generated using `create_sample_data.py`
   - Contains 7 days of synthetic data
   - Sufficient for testing and demonstration

To generate sample data:
```bash
python create_sample_data.py
```

## Requirements

```bash
python == 3.10.4
tensorflow == 2.13.0
pandas >= 2.2.3
numpy >= 2.2.3
scikit-learn
matplotlib
joblib
streamlit >= 1.44.0
```

## Quick Start

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Generate sample data (if you don't have the full dataset):
```bash
python create_sample_data.py
```

3. Launch the web interface:
```bash
python -m streamlit run power_forecast_app.py
```

## Project Structure

- `power_forecast_app.py`: Streamlit web application
- `create_sample_data.py`: Script to generate sample data
- `requirements.txt`: Package dependencies
- `README.md`: Project documentation

## Model Details

### Features Used
- Power demand history
- Time features:
  - Hour of day
  - Day of week
  - Month
  - Weekend flag
- Lag features (5min, 15min, 30min)
- Rolling means (15min, 30min)
- Weather data (temperature, humidity)

### Architecture
```
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32)
Dropout(0.2)
Dense(16, activation='relu')
Dense(1)
```

### Training Parameters
- Sequence length: 30 (2.5 hours)
- Batch size: 32
- Max epochs: 50
- Early stopping patience: 5
- Learning rate reduction on plateau

## Web Interface Features

1. Historical Data View:
   - Last 48 hours visualization
   - Current demand statistics
   - 24-hour average comparison

2. Forecasting:
   - Adjustable forecast horizon (1-24 hours)
   - Real-time predictions
   - Confidence intervals
   - Detailed forecast statistics

3. Analysis Tools:
   - Interactive plots
   - Downloadable forecast data
   - Performance metrics display

## Data Format

Required CSV columns:
```
datetime          - Timestamp (5-minute intervals)
Power demand      - Power demand in MW
Temperature       - Weather data (optional)
Humidity         - Weather data (optional)
[Other weather]  - Additional weather features (optional)
```

## Performance

Typical model performance metrics:
- MAE: 0.0444
- RMSE: 0.0543
- R² Score: 0.9170

## License

MIT License

## Contributors

[Your name/organization]

## Acknowledgments

- Delhi Power Department for data access
- Weather data providers
- Open-source community 
