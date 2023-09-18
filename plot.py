import numpy as np
import matplotlib.pyplot as plt

log_dir = "log_PPO_h1"

npzfile = np.load(log_dir + "/evaluations.npz")

print(npzfile.files)

X = npzfile['timesteps']
Y_mean = np.mean(npzfile['results'],axis=1)
Y_std = np.std(npzfile['results'],axis=1)

print(X)
print(Y_mean)


fig = plt.figure(log_dir)
plt.plot(X, Y_mean)
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.show()