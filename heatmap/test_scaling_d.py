import matplotlib.pyplot as plt
import numpy as np


def logical_error_rate(distance, physical_error_rate, alpha, threshold_error_rate):
    ler = (alpha*physical_error_rate /
           threshold_error_rate)**((distance-1)/2)
    return ler


def main():
    for alpha in range(1, 5):
        for p in np.logspace(-3, -2, 4):

            plt.plot(np.linspace(3, 15, 2), [logical_error_rate(d, p, alpha, 10e-1)
                                             for d in np.linspace(3, 15, 2)], label=f"alpha:{alpha}, p:{p}")


if __name__ == "__main__":
    main()
    plt.legend()
#    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Distance')
    plt.ylabel('Logical Error Rate')
    plt.show()
