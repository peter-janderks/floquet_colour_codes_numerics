#!/usr/bin/env python3

import argparse
import math
import sys
from typing import Optional, List, Any, Tuple, Callable

import sinter
from matplotlib import pyplot as plt



def extrapolate_footprint_achieving_error_rate(
        group: List[sinter.TaskStats],
        *,
        target_p: float,
        failure_unit_func: Callable[[sinter.TaskStats], float],
) -> Optional[sinter.Fit]:
    assert len({stat.json_metadata['p'] for stat in group}) == 1
    sqrt_qs = []
    log_ps = []
    for stat in group:
        if stat.shots:
            p_shot = stat.errors / stat.shots
            if 0 < p_shot < 0.3:
                p_unit = p_shot
                #
                #                p_unit = sinter.shot_error_rate_to_unit_error_rate(
                #                p_unit = xz_piece_error_rate(p_shot, pieces=failure_unit_func(stat), combo=stat.json_metadata['b'] == 'XZ')
                sqrt_qs.append(math.sqrt(stat.json_metadata['distance']**2))
                log_ps.append(math.log(p_unit))

    if len(log_ps) < 2:
        # Can't interpolate a slope from 1 data point.
        return None

    slope_fit = sinter.fit_line_slope(
        xs=log_ps,
        ys=sqrt_qs,
        max_extra_squared_error=1,
    )
    if slope_fit.best >= 0:
        # Slope is going the wrong way! Definitely over threshold.
        return None

    fit = sinter.fit_line_y_at_x(
        xs=log_ps,
        ys=sqrt_qs,
        target_x=math.log(target_p),
        max_extra_squared_error=1,
    )

    return sinter.Fit(
        low=fit.low**2,
        best=fit.best**2,
        high=fit.high**2,
    )


def teraquop_curve(
        group: List[sinter.TaskStats],
        *,
        target_p: float,
        failure_unit_func: Callable[[sinter.TaskStats], float],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs = []
    ys_best = []
    ys_low = []
    ys_high = []
    p_groups = sinter.group_by(group, key=lambda stats: stats.json_metadata['p'])
    for p in sorted(p_groups.keys()):
        p_group = p_groups[p]
        pt = extrapolate_footprint_achieving_error_rate(
            p_group,
            target_p=target_p,
            failure_unit_func=failure_unit_func,
        )
        if pt is not None:
            xs.append(p)
            ys_best.append(pt.best)
            ys_low.append(pt.low)
            ys_high.append(pt.high)
    return xs, ys_low, ys_best, ys_high


def main(csv_file):        
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
#    args.label_func = lambda *, decoder, metadata, strong_id: f"noise={metadata['noise']} basis={metadata['b']} style={metadata['style']}"
#    args.label_func = lambda *, decoder, metadata

 #   if args.order_func is None:
#    args.order_func = args.label_func
    def group_key(stats: sinter.TaskStats) -> Any:
        """the key to use for sinter.group_by
        
        returns a tuple of the output of order_func then label_func
            order_func output should override label in terms of sorting,
            label_func output will be used for the plot label
        
        Note that stats different order outputs but the same label 
        will appear twice in the plot legend with different colors/markers but the same label
        """
        return (
            # ah need to do this
            args.order_func(decoder=stats.decoder, metadata=stats.json_metadata, strong_id=stats.strong_id),
            args.label_func(),
        )

    stats: List[sinter.TaskStats] = sinter.stats_from_csv_files(csv_file)
    target_p = 1e-12
    print('need to fix this')
    #failure_unit_func = lambda stat: stat.json_metadata['total_measurement_rounds']
    failure_unit_func = None

    stats = [
        stat
        for stat in stats
        if stat.json_metadata['noise_model'] == 'SuperconductingNoise' and (stat.json_metadata['r'] in [3,6,6,12])  == True
    ]
    if not stats:
        print(f"WARNING: No stats left after filtering basis in {args.basis} filter_func={args.filter_func_desc}. Skipping plot.", file=sys.stderr)
        return

    markers = "ov*sp^<>8PhH+xXDd|" * 100
    import matplotlib.colors
    colors = list(matplotlib.colors.TABLEAU_COLORS) * 3

    ax: plt.Axes
    fig: plt.Figure
    fig, ax = plt.subplots(1, 1)
    groups = sinter.group_by(stats, key=lambda stats: stats.json_metadata['p'] and stats.json_metadata['code'])
    curves = {
        key: teraquop_curve(
            groups[key],
            target_p=target_p,
            failure_unit_func=failure_unit_func,
        )
        for key in sorted(groups.keys())
    }

    for k, (d, (xs, ys_low, ys_best, ys_high)) in enumerate(curves.items()):
        ax.fill_between(xs, ys_low, ys_high, alpha=0.2, color=colors[k])
    for k, (d, (xs, ys_low, ys_best, ys_high)) in enumerate(curves.items()):
        ax.plot(xs, ys_best, label=d, marker=markers[k], color=colors[k])

    ax.set_title('Teraquop plot')
    ax.set_ylabel(f"Physical qubits to get logical error rate {target_p}")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylim(1e2, 1e6)
    ax.set_xlim(1e-4, 1e-3)
    ax.legend()
    ax.loglog()
    ax.grid(which='minor')
    ax.grid(which='major', color='black')

    fig.set_size_inches(10, 10)
    args.out = 'test.png'
    args.show =True
    if args.out is not None:
        fig.savefig(args.out, bbox_inches='tight', dpi=200)
    if args.show:
        plt.show()


if __name__ == '__main__':
    print('remember to compress first')
    main('scripts/resume_15_2/data.csv')