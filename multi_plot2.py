import numpy as np
import matplotlib.pyplot as plt

def plot(log_dirs):
    fig = plt.figure()

    for log_dir, name in log_dirs:
        npzfile = np.load(log_dir + "/evaluations.npz")

        print(npzfile.files)

        X = npzfile['timesteps']
        Y_mean = np.mean(npzfile['ep_lengths'], axis=1)
        Y_std = np.std(npzfile['ep_lengths'], axis=1) / 5

       
        plt.plot(X, Y_mean, label=name)
        plt.fill_between(X, Y_mean - Y_std, Y_mean + Y_std, alpha=0.3)

    plt.title("Średnia długość epizodu")
    plt.xlabel("Liczba kroków")    
    plt.ylabel("Długość epizodu") 
    # plt.ylim((50,120))
    plt.plot(X, [500]*len(X), linestyle='dashed' )
    plt.legend()
    plt.show()
    # plt.savefig(log_dir + "/wykres.png")

if __name__ == "__main__":
    log_dir = [
        ("log_hgs_mod_PPO_XXX_3", "frame_skip = 4"),
        ("log_hgs_mod_PPO_XXX_4", "frame_skip = 2"),
    ]
    plot(log_dir)