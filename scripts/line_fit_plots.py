from typing import Callable, Sequence, Tuple
import stim
from collections import defaultdict
import math
from random import sample
import sinter
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
from scipy.stats._stats_mstats_common import LinregressResult
from scipy.optimize import leastsq
import matplotlib.transforms as mtransforms


def least_squares_through_point(
    *, xs: np.ndarray, ys: np.ndarray, required_x: float, required_y: float
) -> LinregressResult:
    xs2 = xs - required_x
    ys2 = ys - required_y

    def err(slope: float) -> float:
        return least_squares_cost(xs=xs2, ys=ys2, intercept=0, slope=slope)

    (best_slope,), _ = leastsq(func=err, x0=0.0)
    intercept = required_y - required_x * best_slope
    return LinregressResult(
        best_slope, intercept, None, None, None, intercept_stderr=False
    )


def binary_intercept(
    *,
    func: Callable[[float], float],
    start_x: float,
    step: float,
    target_y: float,
    atol: float,
) -> int:
    """Performs an approximate granular binary search over a monotonically ascending function."""
    start_y = func(start_x)
    if abs(start_y - target_y) <= atol:
        return start_x
    while (func(start_x + step) >= target_y) == (start_y >= target_y):
        step *= 2
        if np.isinf(step) or step == 0:
            raise ValueError("Failed.")
    xs = [start_x, start_x + step]
    min_x = min(xs)
    max_x = max(xs)
    increasing = func(min_x) < func(max_x)

    while True:
        med_x = (min_x + max_x) / 2
        med_y = func(med_x)
        if abs(med_y - target_y) <= atol:
            return med_x
        assert med_x not in [min_x, max_x]
        if (med_y < target_y) == increasing:
            min_x = med_x
        else:
            max_x = med_x


def least_squares_cost(
    *, xs: np.ndarray, ys: np.ndarray, intercept: float, slope: float
) -> float:
    assert len(xs.shape) == 1
    assert xs.shape == ys.shape
    return np.sum((intercept + slope * xs - ys) ** 2)


def least_squares_output_range(
    *, xs: Sequence[float], ys: Sequence[float], target_x: float, cost_increase: float
) -> Tuple[float, float]:
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    fit = linregress(xs, ys)
    base_cost = least_squares_cost(
        xs=xs, ys=ys, intercept=fit.intercept, slope=fit.slope
    )
    base_y = fit.intercept + target_x * fit.slope

    def cost_for_y(y2: float) -> float:
        fit2 = least_squares_through_point(
            xs=xs, ys=ys, required_x=target_x, required_y=y2
        )
        return least_squares_cost(
            xs=xs, ys=ys, intercept=fit2.intercept, slope=fit2.slope
        )

    low_y = binary_intercept(
        start_x=base_y,
        step=-1,
        target_y=base_cost + cost_increase,
        func=cost_for_y,
        atol=1e-5,
    )
    high_y = binary_intercept(
        start_x=base_y,
        step=1,
        target_y=base_cost + cost_increase,
        func=cost_for_y,
        atol=1e-5,
    )
    return low_y, high_y


def distance_to_qubits_normal(distance):
    return (distance**2) + (distance**2 - 1)


def distance_to_qubits_short(distance):
    return (distance**2) + ((distance**2 - 1) / 2)


def multi_plot(plot, p_I_list):
    plot_names = [
        ["$p_I = 0$"],
        ["$p_I = 0.75$"],
        ["$p_I = 0.5$"],
        ["$p_I = 0.25$"],
        ["$p_I = 0.1$"],
        ["$p_I = 0.05$"],
    ]
    fig, axs = plt.subplot_mosaic(plot_names, sharex=True,figsize=(4, 10), constrained_layout=True)
    # constrained_layout=True, figsize = (12,8))
    for index, p_I in enumerate(p_I_list):
        main(plot, p_I, axs[plot_names[index][0]])

    fig.supylabel("Number of qubits")
    axs['$p_I = 0$'].legend(loc="lower right")
    fig.supxlabel("Physical error rate")

    for label, ax in axs.items():
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
    fig.savefig("teraquop_plots.png")
    plt.show()


def main(plot, p_I, ax):
    samples_normal = sinter.stats_from_csv_files(
        f"../data/new_surface_code_p_I_{p_I}.csv"
    )
    # print(samples_normal)
    samples_short = sinter.stats_from_csv_files(
        f"../data/new_short_surface_code_p_I_{p_I}.csv"
    )
    if plot == "line_fit":
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=2, nrows=1, hspace=0.05, wspace=0.05)
        ax = gs.subplots(sharex=True, sharey=True)

        fig, ax = plt.subplots(2, 1)

        line_fit_plot(ax[0], samples_normal)
        line_fit_plot(ax[1], samples_short)
        for axis in ax:
            axis.set_yscale("log")
            axis.grid()
            axis.set_ylabel("Logical Error Probability (per shot)")
            axis.set_xlabel("Distance")
            axis.legend()
            axis.set_xlim([0, 30])
            axis.set_ylim([1e-13, 1e0])
        ax[0].set_title("normal rotated surface code")
        ax[1].set_title("short rotated surface code")
        fig.savefig("line_fit.png")
    elif plot == "lambda":
        ax = plt.axes()

        lambda_plot(ax, samples_normal, 0, "normal sc")
        lambda_plot(ax, samples_short, 1, "short sc")

        ax.loglog()
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Lambda Factor")
        ax.set_xlim(1e-4, 3e-3)
        ax.set_ylim(1, 100)

        ax.grid(which="minor")
        ax.grid(which="major", color="black")

        ax.set_title("Lambda plot")
        #        ax.yaxis.set_ticks_position("both")
        ax.legend()
    elif plot == "teraquop":
        if ax == None:
            ax = plt.axes()
        teraquop_plot(ax, samples_normal, distance_to_qubits_normal, 0, "NMS")
        teraquop_plot(
            ax,
            samples_short,
            distance_to_qubits_short,
            0,
            "SMS",
        )
        # teraquop_plot(
        #    ax, samples_short, distance_to_qubits_normal, 0, "short fixed interactions"
        # )
        #        ax.legend()
        ax.loglog()
        #        ax.set_xlabel("Physical Error Rate")
        #        ax.set_ylabel("Number of qubits")
        ax.grid(which="minor")
        ax.grid(which="major", color="black")



#        ax.set_title("Number of qubits required")


#    if ax == None:
#       plt.show()


def line_fit_plot(ax, data_samples):
    data_to_plot = defaultdict(defaultdict)
    for data in data_samples:
        if data.json_metadata["d"] > 2:
            data_to_plot[data.json_metadata["p"]][data.json_metadata["d"]] = (
                data.errors / data.shots
            )

    colours = ["r", "b", "g", "c", "m", "y", "k", "r", "b", "g"]
    i = 0
    for per in sorted(data_to_plot.keys()):
        ler_at_d = data_to_plot[per]
        distances = []
        lers_filtered = []
        for distance, ler in ler_at_d.items():
            if ler > 0:
                lers_filtered.append(ler)
                distances.append(distance)

        if len(distances) > 1:
            slope, intercept, _, _, _ = linregress(
                distances, y=[math.log(y) for y in lers_filtered]
            )
            if slope < 0:
                ys2 = [1e0, 1e-13]
                xs2 = [(math.log(y) - intercept) / slope for y in ys2]
                distances_for_fit = distances.copy()
                distances_for_fit.append(20)
                ax.plot(xs2, ys2, color=colours[i % len(colours)])

        ax.plot(
            ler_at_d.keys(),
            ler_at_d.values(),
            "x",
            color=colours[i % len(colours)],
            label=f"per = {round(per,6)}",
        )

        i += 1


def get_min_and_max_distances(lers, distances, target_probability):
    d1, d2 = least_squares_output_range(
        xs=distances, ys=lers, target_x=math.log(target_probability), cost_increase=1
    )
    return (int(math.ceil(d1)), int(math.ceil(d2)))


def lambda_plot(ax, data_samples, case_i, name):
    lambda_ys = []
    lambda_ys_low = []
    lambda_ys_high = []
    lambda_xs = []

    data_to_plot = defaultdict(defaultdict)
    for data in data_samples:
        if data.json_metadata["d"] > 2:
            data_to_plot[data.json_metadata["p"]][data.json_metadata["d"]] = (
                data.errors / data.shots
            )

    colours = ["r", "b", "g", "c", "m", "y", "k", "r", "b", "g"]
    i = 0

    for per in sorted(data_to_plot.keys()):
        ler_at_d = data_to_plot[per]
        distances = []
        lers_filtered = []
        for distance, ler in ler_at_d.items():
            if ler > 0:
                lers_filtered.append(ler)
                distances.append(distance)
            elif per == 1e-4:
                print(ler, distance, name)

        if len(distances) > 1:
            sinter_fit = sinter.fit_line_slope(
                xs=distances,
                ys=[math.log(y) for y in lers_filtered],
                max_extra_squared_error=1,
            )
            if sinter_fit.best < 0:
                lambda_ys.append(1 / math.exp(sinter_fit.best) ** 2)
                lambda_ys_low.append(1 / math.exp(sinter_fit.low) ** 2)
                lambda_ys_high.append(1 / math.exp(sinter_fit.high) ** 2)
                lambda_xs.append(per)

    markers = "ov*sp^<>8P+xXDd|"
    ax.plot(lambda_xs, lambda_ys, marker="ov*s"[case_i], label=name)
    ax.fill_between(lambda_xs, lambda_ys_low, lambda_ys_high, alpha=0.3)


def projected_distance(distances, lers, target_error: float) -> float:
    r = linregress(distances, [math.log(y) for y in lers])
    return int(math.ceil((math.log(target_error) - r.intercept) / r.slope))


def teraquop_plot(ax, data_samples, distance_to_qubits, case_i, name):
    data_to_plot = defaultdict(defaultdict)
    for data in data_samples:
        if data.json_metadata["d"] > 2:
            data_to_plot[data.json_metadata["p"]][data.json_metadata["d"]] = (
                data.errors / data.shots
            )

    colours = ["r", "b", "g", "c", "m", "y", "k", "r", "b", "g", "c", "m", "y"]
    i = 0
    teraquop_pers = []
    teraquop_distances = []
    teraquop_min_qubits = []
    teraquop_max_qubits = []
    teraquop_qubits = []

    for per in sorted(data_to_plot.keys()):
        ler_at_d = data_to_plot[per]
        distances = []
        lers_filtered = []
        for distance, ler in ler_at_d.items():
            if ler > 0:
                lers_filtered.append(ler)
                distances.append(distance)
        if len(distances) > 1:

            teraquop_d = projected_distance(distances, lers_filtered, 1e-12)
            if teraquop_d > 0 and per < 0.01:
                d_min, d_max = get_min_and_max_distances(
                    lers_filtered, distances, 1e-12
                )

                teraquop_pers.append(per)
                teraquop_distances.append(teraquop_d)
                teraquop_min_qubits.append(distance_to_qubits(d_min + teraquop_d))
                teraquop_max_qubits.append(distance_to_qubits(d_max + teraquop_d))
                teraquop_qubits.append(distance_to_qubits(teraquop_d))

    markers = "ov*sp^<>8P+xXDd|"
    ax.plot(teraquop_pers, teraquop_qubits, marker="ov*s"[case_i], label=name)
    ax.fill_between(teraquop_pers, teraquop_min_qubits, teraquop_max_qubits, alpha=0.3)


#    ax.set_ylabel("Number of qubits")
if __name__ == "__main__":

    multi_plot("teraquop", [1.0, 0.75, 0.5, 0.25, 0.1, 0.05])
# 5
