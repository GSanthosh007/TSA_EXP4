# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 11-05-2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 500 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 500 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.
6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.
### PROGRAM:
```
#Ex-04

# Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('dataset.csv')


# Use all samples from dataset
X = data['temp_mean(c)'][0:500]

# Total number of samples
N = len(X)

# Original Data Visualization
plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(X)
plt.title('Original Data')
plt.show()

# ACF and PACF of Original Data
plt.subplot(2, 1, 1)
plot_acf(X, lags=50, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=50, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# ARMA(1,1) Model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1)
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()

# ARMA(2,2) Model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2)
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()
```
OUTPUT:

ORIGINAL DATA:
<img width="962" height="495" alt="image" src="https://github.com/user-attachments/assets/87ec77fd-73f6-4cc7-87ac-6ec0768e8ff8" />

SIMULATED ARMA(1,1) PROCESS:
<img width="1012" height="500" alt="image" src="https://github.com/user-attachments/assets/64d4b35c-6128-43f4-be60-2d5553076649" />



PARTIAL AUTOCORRELATION:
<img width="1042" height="503" alt="image" src="https://github.com/user-attachments/assets/5326c94c-bc58-48b4-9fa8-245f19285d53" />


AUTOCORRELATION:
<img width="1058" height="492" alt="image" src="https://github.com/user-attachments/assets/b4c65596-810d-4d6e-b3ce-1b84812b4219" />



SIMULATED ARMA(2,2) PROCESS:
<img width="991" height="502" alt="image" src="https://github.com/user-attachments/assets/72aa8aa5-9669-4de9-ad8c-217584af1a8c" />

PARTIAL AUTOCORRELATION:

<img width="1052" height="492" alt="image" src="https://github.com/user-attachments/assets/e7dc28f7-3b01-4296-80c1-eb741f151501" />


AUTOCORRELATION:
<img width="977" height="506" alt="image" src="https://github.com/user-attachments/assets/45fef5cf-5e66-41ee-878b-61e42aff05a4" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
