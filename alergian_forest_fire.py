# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import codecademylib3

# Load data
forests = pd.read_csv('forests.csv')

# =============================================================================
# TASK 1: Check multicollinearity with correlation heatmap
# =============================================================================
# Create correlation table for quantitative variables
corr_grid = forests.corr()

# Plot heatmap
sns.heatmap(corr_grid, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Forest Variables')
plt.show()
plt.clf()

# =============================================================================
# TASK 2: Scatter plot of humidity vs temperature, colored by region
# =============================================================================
bejaia = forests[forests['region'] == 'Bejaia']
sidi = forests[forests['region'] == 'Sidi Bel-abbes']

plt.scatter(bejaia['temp'], bejaia['humid'], c='blue', label='Bejaia', alpha=0.6)
plt.scatter(sidi['temp'], sidi['humid'], c='red', label='Sidi Bel-abbes', alpha=0.6)
plt.title('Humidity vs Temperature by Region')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Humidity (%)')
plt.legend()
plt.show()
plt.clf()

# =============================================================================
# TASK 3: Multiple linear regression predicting humidity
# =============================================================================
# Create dummy variable for region (Sidi Bel-abbes = 1, Bejaia = 0)
forests['region_dummy'] = forests['region'].map({'Bejaia': 0, 'Sidi Bel-abbes': 1})

# Define predictors and outcome (use add_constant for intercept)
X_humidity = forests[['temp', 'region_dummy']]
X_humidity = sm.add_constant(X_humidity)  # Add intercept term
y_humidity = forests['humid']

# Fit model
modelH = sm.OLS(y_humidity, X_humidity).fit()
print(modelH.params)

# Access coefficients using column names (not index numbers)
intercept = modelH.params['const']        # or modelH.params[0]
temp_coef = modelH.params['temp']          # or modelH.params[1]
region_coef = modelH.params['region_dummy']  # or modelH.params[2]

# =============================================================================
# TASK 4: Write regression equations
# =============================================================================
# Full regression equation (as comment):
# humid = β₀ + β₁(temp) + β₂(region_dummy)
# Where β₀ = intercept, β₁ = temp coefficient, β₂ = region coefficient

# For Bejaia (region_dummy = 0):
# humid_Bejaia = β₀ + β₁(temp)

# For Sidi Bel-abbes (region_dummy = 1):
# humid_Sidi = β₀ + β₁(temp) + β₂
# humid_Sidi = (β₀ + β₂) + β₁(temp)

# =============================================================================
# TASK 5: Interpretations
# =============================================================================
# Coefficient on temp interpretation (as comment):
# "The coefficient on temp represents the change in relative humidity for each 1°C 
# increase in temperature, holding region constant."

# Bejaia intercept interpretation (as comment):
# "For Bejaia, the intercept represents the predicted humidity when temp = 0°C."

# Sidi Bel-abbes intercept interpretation (as comment):
# "For Sidi Bel-abbes, the intercept is (β₀ + β₂), representing predicted humidity 
# when temp = 0°C for this region."

# =============================================================================
# TASK 6: Scatter plot with regression lines
# =============================================================================
bejaia = forests[forests['region'] == 'Bejaia']
sidi = forests[forests['region'] == 'Sidi Bel-abbes']

plt.scatter(bejaia['temp'], bejaia['humid'], c='blue', label='Bejaia', alpha=0.6)
plt.scatter(sidi['temp'], sidi['humid'], c='red', label='Sidi Bel-abbes', alpha=0.6)

# Plot regression lines
temp_range = np.array([forests['temp'].min(), forests['temp'].max()])

# Bejaia line (region_dummy = 0)
humid_Bejaia = intercept + temp_coef * temp_range
plt.plot(temp_range, humid_Bejaia, color='blue', label='Bejaia regression line', linewidth=2)

# Sidi Bel-abbes line (region_dummy = 1)
humid_Sidi = (intercept + region_coef) + temp_coef * temp_range
plt.plot(temp_range, humid_Sidi, color='red', label='Sidi Bel-abbes regression line', linewidth=2)

plt.title('Humidity vs Temperature with Regression Lines by Region')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Humidity (%)')
plt.legend()
plt.show()
plt.clf()

# =============================================================================
# TASK 7: Scatter plot of FFMC vs temperature, colored by fire status
# =============================================================================
no_fire = forests[forests['fire'] == False]
fire = forests[forests['fire'] == True]

plt.scatter(no_fire['temp'], no_fire['FFMC'], c='blue', label='No Fire', alpha=0.6)
plt.scatter(fire['temp'], fire['FFMC'], c='red', label='Fire', alpha=0.6)
plt.title('FFMC vs Temperature by Fire Status')
plt.xlabel('Temperature (°C)')
plt.ylabel('Fine Fuel Moisture Code (FFMC)')
plt.legend()
plt.show()
plt.clf()

# =============================================================================
# TASK 8: Model predicting FFMC with interaction term
# =============================================================================
forests['fire_dummy'] = forests['fire'].map({False: 0, True: 1})
forests['temp_fire_interaction'] = forests['temp'] * forests['fire_dummy']

X_ffmc = forests[['temp', 'fire_dummy', 'temp_fire_interaction']]
X_ffmc = sm.add_constant(X_ffmc)
y_ffmc = forests['FFMC']

resultsF = sm.OLS(y_ffmc, X_ffmc).fit()
print(resultsF.params)

# Access coefficients using names
beta0 = resultsF.params['const']
beta1 = resultsF.params['temp']
beta2 = resultsF.params['fire_dummy']
beta3 = resultsF.params['temp_fire_interaction']

# =============================================================================
# TASK 9: Write regression equations
# =============================================================================
# Full equation (as comment):
# FFMC = β₀ + β₁(temp) + β₂(fire_dummy) + β₃(temp × fire_dummy)

# For locations without fire (fire_dummy = 0):
# FFMC_no_fire = β₀ + β₁(temp)

# For locations with fire (fire_dummy = 1):
# FFMC_fire = (β₀ + β₂) + (β₁ + β₃)(temp)

# =============================================================================
# TASK 10: Interpret temp coefficient for each group
# =============================================================================
# For locations without fire (as comment):
# "For locations without fire, the coefficient on temp is β₁."

# For locations with fire (as comment):
# "For locations with fire, the coefficient on temp is β₁ + β₃."

# =============================================================================
# TASK 11: Plot with interaction regression lines
# =============================================================================
no_fire = forests[forests['fire'] == False]
fire = forests[forests['fire'] == True]

plt.scatter(no_fire['temp'], no_fire['FFMC'], c='blue', label='No Fire', alpha=0.6)
plt.scatter(fire['temp'], fire['FFMC'], c='red', label='Fire', alpha=0.6)

temp_range = np.array([forests['temp'].min(), forests['temp'].max()])

# No fire line
ffmc_no_fire = beta0 + beta1 * temp_range
plt.plot(temp_range, ffmc_no_fire, color='blue', label='No fire regression line', linewidth=2)

# Fire line
ffmc_fire = (beta0 + beta2) + (beta1 + beta3) * temp_range
plt.plot(temp_range, ffmc_fire, color='red', label='Fire regression line', linewidth=2)

plt.title('FFMC vs Temperature with Interaction Regression Lines')
plt.xlabel('Temperature (°C)')
plt.ylabel('Fine Fuel Moisture Code (FFMC)')
plt.legend()
plt.show()
plt.clf()

# =============================================================================
# TASK 12: Scatter plot of FFMC vs relative humidity
# =============================================================================
plt.scatter(forests['humid'], forests['FFMC'], alpha=0.6)
plt.title('FFMC vs Relative Humidity')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Fine Fuel Moisture Code (FFMC)')
plt.show()
plt.clf()

# =============================================================================
# TASK 13: Polynomial model predicting FFMC from humid
# =============================================================================
forests['humid_squared'] = forests['humid'] ** 2

X_poly = forests[['humid', 'humid_squared']]
X_poly = sm.add_constant(X_poly)
y_ffmc2 = forests['FFMC']

resultsP = sm.OLS(y_ffmc2, X_poly).fit()
print(resultsP.params)

beta0_p = resultsP.params['const']
beta1_p = resultsP.params['humid']
beta2_p = resultsP.params['humid_squared']

# =============================================================================
# TASK 14: Write equation and calculate sample predictions
# =============================================================================
# FFMC = β₀ + β₁(humid) + β₂(humid²)

humidity_levels = [25, 35, 60, 70]
for h in humidity_levels:
    ffmc = beta0_p + beta1_p * h + beta2_p * (h ** 2)
    print(f"Humidity {h}%: FFMC = {ffmc:.2f}")

# =============================================================================
# TASK 15: Interpret relationship
# =============================================================================
# (Add your interpretation as comments in the code)

# =============================================================================
# TASK 16: Multiple regression with all four variables + FWI model
# =============================================================================
X_ffmc_all = forests[['humid', 'temp', 'wind', 'rain']]
X_ffmc_all = sm.add_constant(X_ffmc_all)
y_ffmc_all = forests['FFMC']

results_all = sm.OLS(y_ffmc_all, X_ffmc_all).fit()
print("FFMC model with all 4 variables:")
print(results_all.params)

# Predict FWI from ISI and BUI
X_fwi = forests[['ISI', 'BUI']]
X_fwi = sm.add_constant(X_fwi)
y_fwi = forests['FWI']

results_fwi = sm.OLS(y_fwi, X_fwi).fit()
print("\nFWI model from ISI and BUI:")
print(results_fwi.params)
