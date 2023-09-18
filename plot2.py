import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log_dir = "log_basic_PPO_3"

npzfile = np.load(log_dir + "/evaluations.npz")

print(npzfile.files)

X = npzfile['timesteps']
Y_mean = np.mean(npzfile['results'], axis=1)
Y_std = np.std(npzfile['results'], axis=1)

# Calculate moving average using pandas rolling function
window_size = 10  # Set the window size for the moving average
Y_mean_moving_avg = pd.Series(Y_mean).rolling(window_size, min_periods=1).mean()

fig = plt.figure(log_dir)
plt.plot(X, Y_mean_moving_avg, label='Średnia nagroda')
plt.fill_between(X, Y_mean - Y_std, Y_mean + Y_std, alpha=0.3, label='Odchylenie standardowe')
plt.xlabel("Liczba kroków")
plt.ylabel("Nagrody")
plt.legend()
plt.show()
