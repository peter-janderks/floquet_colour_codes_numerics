import sinter
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit


def get_thresholds():
    normal_thresholds = []
    short_thresholds = []
    idling_error_rates = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
"""    for p_I in idling_error_rates:
        samples_normal = sinter.stats_from_csv_files(
            "../data/new_surface_code_p_I_" + str(p_I) + ".csv"
        )
        normal_thresholds.append(get_one_threshold(samples_normal, p_I))
        samples_short = sinter.stats_from_csv_files(
            "../data/new_short_surface_code_p_I_" + str(p_I) + ".csv"
        )
        short_thresholds.append(get_one_threshold(samples_short, p_I))

    plt.plot(idling_error_rates, normal_thresholds, "*", label="NMS")
    plt.plot(idling_error_rates, short_thresholds, "x", label="SMS")
    plt.legend()
    plt.ylabel("Error threshold")
    plt.xlabel("Idling error probability")
    plt.grid()
    plt.savefig("thresholds_vs_pidling.png")
    plt.show()
"""

def get_one_threshold(samples_normal, p_I, min_per, max_per, min_dis):
    nms_data = {}

    for sample in samples_normal:
        #       print(sample.__dir__())
        if sample.json_metadata["distance"] not in nms_data:
            nms_data[sample.json_metadata["distance"]] = {}
        #        print(sample.errors)
        nms_data[sample.json_metadata["distance"]][
            sample.json_metadata["p"]
        ] = sample.errors / (sample.shots - sample.discards)

    nms_distances = []
    nms_pers = []
    nms_lers = []
    for d in nms_data:
        if d > min_dis:
            for per in nms_data[d]:
                if nms_data[d][per] < 0.4 and per>min_per and per<max_per:
                    nms_distances.append(d)
                    nms_pers.append(per)
                    nms_lers.append(nms_data[d][per])

    return calc_threshold(nms_pers, nms_distances, nms_lers)

    #    print(data)


def main():
    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(2, 1)
    samples_normal = sinter.stats_from_csv_files("../data/FloquetColourCode_bias_2.csv")
    samples_short = sinter.stats_from_csv_files(
        "../data/FloquetColourCode_bias_2.csv"
    )
    normal_threshold = get_one_threshold(samples_normal, 1,0.003,0.0045)
    short_threshold = get_one_threshold(samples_short, 1)
    sinter.plot_error_rate(
        ax=ax[1],
        stats=samples_short,
        group_func=lambda stat: f"d={stat.json_metadata['distance']}",
        x_func=lambda stat: stat.json_metadata["p"],
    )

    sinter.plot_error_rate(
        ax=ax[0],
        stats=samples_normal,
        group_func=lambda stat: f"d={stat.json_metadata['distance']}",
        x_func=lambda stat: stat.json_metadata["p"],
    )
    #    plt.plot(pers,lers)
    #    plt.show()
    ax[0].axvline(x=normal_threshold, color="black", linestyle="dashed")
    ax[1].axvline(x=short_threshold, color="black", linestyle="dashed")
    for axis in ax:
        axis.loglog()
        #    ax.set_ylim(1e-5, 1)
        axis.grid()
        axis.set_ylabel("Logical error probability")

        axis.legend()
    #        axis.plot(error_probabilities, error_probabilities)
    ax[1].set_xlabel("Physical error rate")
    ax[0].set_title("Normal measurement scheme")
    ax[1].set_title("Short measurement scheme")
    #    ax[2].set_title("short rotated surface code with idling noise")
    fig.tight_layout()
    fig.savefig("threshold_pI_1.png")

    plt.show()


def calc_threshold(per_data, distance_data, ler_data):
    print(distance_data)
    print(per_data)
    print(ler_data)
    popt, pcov = curve_fit(
        threshold_fit, (per_data, distance_data), ler_data, maxfev=10000
    )
    return popt[-1]


def threshold_fit(variables, B0, B1, B2, mu, pth):
    p, L = variables
    return (
        B0 + B1 * (p - pth) * pow(L, 1 / mu) + B2 * pow((p - pth) * pow(L, 1 / mu), 2)
    )


if __name__ == "__main__":
#    get_thresholds()
    main()
