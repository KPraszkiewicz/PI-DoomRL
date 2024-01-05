import numpy as np
import matplotlib.pyplot as plt

def plot(log_dir):
    npzfile = np.load(log_dir + "/evaluations.npz")

    print(npzfile.files)

    X = npzfile['timesteps']
    Y_mean = np.mean(npzfile['ep_lengths'], axis=1)
    Y_std = np.std(npzfile['ep_lengths'], axis=1)

    fig = plt.figure(log_dir)
    plt.plot(X, Y_mean, label='średnia długość epizodu')
    plt.fill_between(X, Y_mean - Y_std, Y_mean + Y_std, alpha=0.3)
    plt.xlabel("Liczba kroków")
    plt.ylabel("Długość epizodu")
    # plt.ylim((50,120))
    # plt.plot(X, [90]*len(X), linestyle='dashed' )
    plt.legend()
    plt.savefig(log_dir + "/wykres.png")

if __name__ == "__main__":
    log_dir = "log_hgs_mod_PPO_XXX_4"
    plot(log_dir)