import numpy as np
import matplotlib.pyplot as plt

def plot(log_dirs):
    fig = plt.figure()

    for log_dir, name in log_dirs:
        npzfile = np.load(log_dir + "/evaluations.npz")

        print(npzfile.files)

        X = npzfile['timesteps']
        Y_mean = np.mean(npzfile['results'], axis=1)
        Y_std = np.std(npzfile['results'], axis=1) / 5

       
        plt.plot(X, Y_mean, label=name)
        plt.fill_between(X, Y_mean - Y_std, Y_mean + Y_std, alpha=0.3)


    plt.xlabel("Liczba kroków")    
    plt.ylabel("Nagrody") 
    # plt.ylim((50,120))
    plt.plot(X, [10]*len(X), linestyle='dashed' )
    plt.legend()
    plt.show()
    # plt.savefig(log_dir + "/wykres.png")

if __name__ == "__main__":
    log_dir = [
        ("log_deathmatch_simple_PPO_final_1", "base"),
        ("log_deathmatch_simple_PPO_final_2", "Reward shaping"),
        ("log_deathmatch_simple_PPO_final_4", "Custom net + Reward shaping")
    ]
    plot(log_dir)