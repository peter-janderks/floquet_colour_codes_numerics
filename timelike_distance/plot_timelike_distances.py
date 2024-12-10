import itertools
from main.codes.tic_tac_toe.gauge.GaugeHoneycombCode import GaugeHoneycombCode
from main.codes.tic_tac_toe.gauge.GaugeFloquetColourCode import GaugeFloquetColourCode
from main.building_blocks.detectors.Stabilizer import Stabilizer
from main.compiling.compilers.AncillaPerCheckCompiler import AncillaPerCheckCompiler
from main.compiling.noise.models import PhenomenologicalNoise
from main.compiling.syndrome_extraction.extractors.ancilla_per_check.mixed.CxCyCzExtractor import CxCyCzExtractor
from matplotlib import pyplot as plt
from typing import Literal
import stim


def generate_circuit(code: Literal['HoneycombCode'],
                     rounds: int,
                     distance: int,
                     observable_type: str = 'X',
                     pauli_noise_probability: float = 0.1,
                     measurement_noise_probability: float = 0.1) -> stim.Circuit:
    """Generates a quantum error correction circuit for the Honeycomb code.

    Args:
        rounds (int): The number of rounds for the circuit.
        distance (int): The distance of the Honeycomb code.
        observable_type (str): The type of logical observable ('X' or 'Z'). Defaults to 'X'.

    Returns:
        Any: The compiled stim circuit.
    """

    if observable_type == 'X':
        logical_observables = [code.x_stability_operator]

        initial_stabilizers = [Stabilizer([(0, check)], 0)
                               for check in code.check_schedule[code.gauge_factors[0] + code.gauge_factors[1]]]
        final_measurements = code.get_possible_final_measurement(
            code.logical_qubits[1].z, rounds)

    elif observable_type == 'Z':
        logical_observables = [code.z_stability_operator]
        initial_stabilizers = [Stabilizer([(0, check)], 0)
                               for check in code.check_schedule[0]]
        final_measurements = code.get_possible_final_measurement(
            code.logical_qubits[0].x, rounds)

    noise_model = PhenomenologicalNoise(
        pauli_noise_probability, measurement_noise_probability)

    compiler = AncillaPerCheckCompiler(
        noise_model=noise_model,
        syndrome_extractor=CxCyCzExtractor()
    )
    stim_circuit = compiler.compile_to_stim(
        code=code,
        total_rounds=rounds,
        initial_stabilizers=initial_stabilizers,
        observables=logical_observables,
        final_measurements=final_measurements
    )

    return stim_circuit


def get_graphlike_distance(circuit: stim.Circuit) -> int:
    return len(circuit.detector_error_model(
        approximate_disjoint_errors=True).shortest_graphlike_error())


def get_hyper_edge_distance(circuit: stim.Circuit) -> int:

    logical_errors = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6,
        dont_explore_edges_increasing_symptom_degree=False,
    )

    return len(logical_errors)


def get_distances(code: str, observable_type: str, distances: dict):
    if isinstance(code, GaugeHoneycombCode):
        dict_key = f"{type(code).__name__}_{code.gauge_factors}_{observable_type}"
        distances[dict_key] = dict(
        )
    elif isinstance(code, GaugeFloquetColourCode):
        dict_key = f"{type(code).__name__}_{code.x_gf}_{code.z_gf}_{observable_type}"

        distances[dict_key] = dict()
    for n_rounds in range(12, 25):
        if isinstance(code, GaugeHoneycombCode):
            #            stim_circuit = generate_circuit(
         #               code, n_rounds, 4, observable_type, 0, 0.1)
            distances[dict_key][n_rounds] = code.get_graphlike_timelike_distance(
                n_rounds, observable_type)
#            distances[dict_key][n_rounds] = get_graphlike_distance(
 #               stim_circuit)
        elif isinstance(code, GaugeFloquetColourCode):
            stim_circuit = generate_circuit(
                code, n_rounds, 4, observable_type, 0, 0.1)
            distances[dict_key][n_rounds] = get_graphlike_distance(
                stim_circuit)
#            if observable_type == 'X':
 #               distances[dict_key][n_rounds] = code.get_measurement_error_distance(
  #                  n_rounds, 'X')
   #         elif observable_type == 'Z':
    #            distances[dict_key][n_rounds] = code.get_measurement_error_distance(
     #               n_rounds, 'Z')
    return distances


def main():
    distances = dict()
    distances["X"] = dict()
    distances["Z"] = dict()
    """
    for gauge_factors in itertools.product([1, 2, 3], repeat=3):
        distances["X"] = get_distances(
            GaugeHoneycombCode(4, gauge_factors), "X", distances["X"])
        distances["Z"] = get_distances(
            GaugeHoneycombCode(4, gauge_factors), "Z", distances["Z"])

    for gauge_factors in itertools.product([1, 2, 3], repeat=2):
        distances["X"] = get_distances(
            GaugeFloquetColourCode(4, gauge_factors), "X", distances["X"])
        distances["Z"] = get_distances(
            GaugeFloquetColourCode(4, gauge_factors), "Z", distances["Z"])
    """
    distances["X"] = get_distances(
        GaugeHoneycombCode(4, [3, 1, 1]), "X", distances["X"])
    distances["X"] = get_distances(
        GaugeFloquetColourCode(4, [3, 1, 0]), "X", distances["X"])
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    shape_dict = {'GaugeHoney': 'o', 'GaugeFloqu': 'x'}

    for key, value in distances["X"].items():
        ax2.plot(value.keys(), value.values(),
                 label=key,
                 marker=shape_dict[key[:10]])

#    for key, value in distances["Z"].items():
 #       ax2.plot(value.keys(), value.values(),
  #               label=key, marker=shape_dict[key[:10]])

    ax1.set_xlabel('Number of rounds')
    ax1.set_ylabel('Timelike distance')
    ax2.set_xlabel('Number of rounds')

    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


if __name__ == "__main__":
    main()
