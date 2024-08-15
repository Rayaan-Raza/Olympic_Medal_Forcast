# %%
import numpy as np
import matplotlib as plt
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM



# %%
df = pd.read_csv("../data/Summer-Olympic-medals-1976-to-2008.csv", encoding='ISO-8859-1')


# %%
df.shape

# %%
null_counts = df.isnull().sum()
print(null_counts)

# %%
df.info()


# %%
null_columns = df.columns[df.isnull().any()]
print(null_columns)

# %%
# Step 1: Filter the DataFrame to include only gold medals
gold_medals = df[df['Medal'] == 'Gold']

# Step 2: Group by 'Country' and 'Year', then count the number of medals
gold_medal_counts = gold_medals.groupby(['Country', 'Year']).size().unstack(fill_value=0)

# Step 3: Plot the line chart
plt.figure(figsize=(12, 5))
for country in gold_medal_counts.index:
    plt.plot(gold_medal_counts.columns, gold_medal_counts.loc[country], label=country)

plt.xlabel('Year')
plt.ylabel('Number of Gold Medals')
plt.title('Number of Gold Medals by Country Over Time')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
plt.grid(True)
plt.show()

# %%
# Group by Country and Year to get the total number of medals each country won each year
country_medal_counts = df.groupby(['Country', 'Year']).size().unstack(fill_value=0)


print(country_medal_counts)


# %%
total_medals_by_country = df.groupby('Country').size()

total_medals_by_country = total_medals_by_country.sort_values(ascending=False)

plt.figure(figsize=(20, 5))
total_medals_by_country.plot(kind='bar', color='skyblue')

plt.xlabel('Country')
plt.ylabel('Total Number of Medals')
plt.title('Total Number of Medals by Country (1976-2008)')
plt.xticks(rotation=90)  
plt.grid(axis='y')

# Show the plot
plt.show()

# %%
cutoff_year = 1996


# %%
# Prepare the data for a specific country, e.g., 'USA'
country = "United States"
country_data = df[df['Country'] == country].groupby('Year').size().reset_index(name='y')
country_data.columns = ['ds', 'y']
country_data['ds'] = pd.to_datetime(country_data['ds'], format='%Y')
print(country_data)



# %%


# %%
# Split the data into training and testing sets
train_data = country_data[country_data['ds'].dt.year <= cutoff_year]
test_data = country_data[country_data['ds'].dt.year > cutoff_year]

# %%
print(train_data.shape)
print(train_data.head())

# %%
if not train_data.empty:
    # Initialize and fit the model
    model = Prophet(yearly_seasonality=True)
    model.fit(train_data)

    # Proceed with forecasting
else:
    print("Insufficient data for the selected country and cutoff year.")

# %%
# Define the future period for which you want to make predictions (e.g., next 5 Olympics)
future_years = 5
future = model.make_future_dataframe(periods=future_years * 4, freq='Y')  # Assuming Olympics every 4 years

# Make predictions
forecast = model.predict(future)

# %%
# Merge the actual and predicted data for comparison
comparison = pd.merge(test_data, forecast[['ds', 'yhat']], how='left', on='ds')

# Display the comparison
print(comparison)

# %%
# Plot the predictions along with the actual data
plt.figure(figsize=(12, 6))

# Plot actual data
plt.plot(country_data['ds'], country_data['y'], label='Actual', marker='o')

# Plot forecast
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')

# Highlight the test period
plt.axvspan(test_data['ds'].min(), test_data['ds'].max(), color='gray', alpha=0.3)

plt.xlabel('Year')
plt.ylabel('Number of Medals')
plt.title(f'Forecasting Olympic Medals for {country}')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Predict medals for the next Olympics
future_medals = model.predict(future)

# Extract relevant predictions
future_medals = future_medals[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Display the forecasted medals
print(future_medals.tail(future_years))

# %%
plt.figure(figsize=(12, 6))

# Plot actual historical data
plt.plot(country_data['ds'], country_data['y'], label='Actual', marker='o')

# Plot forecasted data
plt.plot(future_medals['ds'], future_medals['yhat'], label='Forecast', linestyle='--')

# Add uncertainty intervals to the forecast
plt.fill_between(future_medals['ds'], future_medals['yhat_lower'], future_medals['yhat_upper'], color='gray', alpha=0.2)

plt.xlabel('Year')
plt.ylabel('Number of Medals')
plt.title(f'Olympic Medal Predictions for {country}')
plt.legend()
plt.grid(True)
plt.show()

# %%
import pickle

# Save the model
with open('prophet_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# %%



