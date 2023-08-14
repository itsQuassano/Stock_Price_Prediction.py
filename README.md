# Stock_Price_Prediction.py
📈 Stock Market Data Magic
Welcome to the Stock Market Data Magic repository! Here, we'll uncover the hidden insights of stock data using Python spells and enchanting libraries. 🧙‍♂️

🪄 Casting the Spells
To cast our magical data spells, we've gathered a team of powerful libraries:

Python Pandas Matplotlib Seaborn Sklearn XGBoost

📚 Unveiling the Code Chronicles
Behold the secrets of the code incantations:


# Import the magical spells
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Load the mystical data
df = pd.read_excel("path_to_your_excel_file.xlsx", sheet_name="Worksheet")

# 📊 Visualizing the Market Spirits
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')

# More Enchanting Visuals
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.distplot(df[col])

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.boxplot(df[col])

# 🌌 Time and Space Manipulation
splitted = df['Date'].str.split('-', expand=True)
df['day'] = splitted[2].astype('int')
df['month'] = splitted[1].astype('int')
df['year'] = splitted[0].astype('int')

# 🔮 The Quarter End Prediction
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
🧐 Exploring the Insights
Our spells unearth incredible insights:

🔍 Yearly Aggregation: Grouping the data by year, we unlock the magic of yearly averages.

📊 Bar Chart Conjuring: Through bar charts, we reveal the secrets of 'Open', 'High', 'Low', and 'Close' prices.

🥧 Pie of Destiny: A pie chart showcases the distribution of price change predictions.

🎭 Correlation Heatmap: The correlation heatmap unmasks relationships between features.

🪅 Enchanted Results
The enchanted code brings to life a symphony of numbers and patterns, guiding us through the mystical world of stock data. May your quest for knowledge be fruitful, and your journey through data be as mesmerizing as the stars in the night sky. ✨
