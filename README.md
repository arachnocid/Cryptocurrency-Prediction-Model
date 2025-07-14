# **Cryptocurrency Prediction Model**
This repository is my data science project focused on forecasting **POSSIBLE** future changes in cryptocurrency prices.

### Google Colab Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Uwk2WeW7bd0F0wqPSwxGyl_U5KqKPm7P?usp=sharing)

# **Key Features**
## 1. Machine Learning Modeling:

Two main modeling approaches are utilized - a **_Sequential_** neural network model for price prediction based on specific features, and a **_VARMAX_** statistical model for time series forecasting to estimate potential future values of those features.

#### Sequential Model Architecture
<img src="https://github.com/arachnocid/Cryptocurrency-Prediction-Model/blob/main/model_architecture.png">

## 2. Interactive Visualization:

An interactive chart is implemented to display both the observed and forecasted cryptocurrency prices, allowing users to access date and price by moving a cursor.

# **Prerequisites**

### Libraries
- re
- numpy
- pandas
- matplotlib
- bokeh
- sklearn
- keras
- statsmodels

### Data
- Cryptocurrency historical data was collected from **_investing.com_** to gather a comprehensive dataset of Bitcoin exchange rates over time.
- The file includes various features such as open, high, low, close prices, trading volume, and percentage changes.
- Any historical cryptocurrency data from this site can be used in the model.

# **Getting Started**
### Obtaining the data:

- You can download data of a variety of cryptocurrencies from **_investing.com_** website.
- Bitcoin data can be obtained from the following link: https://www.investing.com/crypto/bitcoin/historical-data

### Running from Source Code:
1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/arachnocid/Cryptocurrency-Prediction-Model.git
2. Navigate to the project directory.
   ```bash
   cd Cryptocurrency-Prediction-Model
3. Install the required dependencies (see requirements.txt)
   ```bash
   pip install -r requirements.txt
4. Run the "Cryptocurrency-Prediction-Model.py" script.

## License
This project is licensed under the GPL-3.0 license.

## Author
Arachnocid
