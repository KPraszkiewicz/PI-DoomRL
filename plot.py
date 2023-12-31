import numpy as np
import matplotlib.pyplot as plt

def plot(log_dir):
    npzfile = np.load(log_dir + "/evaluations.npz")

    print(npzfile.files)

    X = npzfile['timesteps']
    Y_mean = np.mean(npzfile['results'], axis=1)
    Y_std = np.std(npzfile['results'], axis=1)

    fig = plt.figure(log_dir)
    plt.plot(X, Y_mean, label='Średnia nagroda')
    plt.fill_between(X, Y_mean - Y_std, Y_mean + Y_std, alpha=0.3, label='Odchylenie standardowe')
    plt.xlabel("Liczba kroków")
    plt.ylabel("Nagrody")
    plt.legend()
    plt.savefig(log_dir + "/wykres.png")

if __name__ == "__main__":
    log_dir = "log_health_gathering_supreme_PPO_PICKUP_4"
    plot(log_dir)