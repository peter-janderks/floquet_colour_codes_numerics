import hashlib
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable
import numpy as np
import stim
from scipy.optimize import curve_fit

import sys
import sinter
from matplotlib import pyplot as plt

from main.building_blocks.pauli import Pauli
from main.building_blocks.pauli.PauliLetter import PauliLetter
from main.codes.tic_tac_toe.FloquetColourCode import FloquetColourCode
from main.codes.tic_tac_toe.HoneycombCode import HoneycombCode
from main.codes.tic_tac_toe.TicTacToeCode import TicTacToeCode
from main.compiling.compilers.AncillaPerCheckCompiler import AncillaPerCheckCompiler
from main.compiling.noise.models import (
    PhenomenologicalNoise,
    CircuitLevelNoise,
    CodeCapacityNoise,
)
from main.compiling.syndrome_extraction.extractors.ancilla_per_check.mixed.CxCyCzExtractor import (
    CxCyCzExtractor,
)
from main.building_blocks.detectors.Stabilizer import Stabilizer
from main.utils.enums import State
from main.utils.utils import output_path

from main.codes.tic_tac_toe.gauge_honeycomb_code import GaugeHoneycombCode
from main.codes.tic_tac_toe.gauge_floquet_colour_code import GaugeFloquetColourCode
def get_bias_tasks(
    constructor: Callable[[int], TicTacToeCode], code_name, ps, bias, bias_type, distances
):
    #    biases = [1/8, 1/4, 1/2, 1, 2, 4, 8]

    syndrome_extractor = CxCyCzExtractor()
    # We'll first compile to our own Circuit class to calculate the number
    # of locations Q where noise could be added. Then we'll use this Q to
    # calculate m and q, and then compile to a stim.Circuit using these
    # parameters.
    pre_compiler = AncillaPerCheckCompiler(
        noise_model=PhenomenologicalNoise(1,  1),
        syndrome_extractor=syndrome_extractor,
    )

    tasks = defaultdict(list)
    for bias in biases:
        print(f"Bias = {bias}")
        for distance in distances:
            print(f"Distance = {distance}")
           
            if code_name == "HoneycombCode":
                layers = 2 * distance
            else:
                layers = distance

            if bias_type == "measurement_vs_data_qubit":
                if code_name == "Gauge2HoneycombCode" or "Gauge2FloquetColourCode":
                    code = constructor(distance,2)
                elif code_name == "Gauge3HoneycombCode" or "Gauge3FloquetColourCode":
                    code = constructor(distance,3)

                else:
                    code = constructor(distance)

                logical_observables = [code.logical_qubits[1].x]

                data_qubits = code.data_qubits.values()
                final_measurements = [
                    Pauli(qubit, PauliLetter("X")) for qubit in data_qubits]  
                initial_stabilizers = []
                for check in code.check_schedule[0]:
                    initial_stabilizers.append(Stabilizer([(0,check)],0))
                pre_circuit: stim.Circuit = pre_compiler.compile_to_circuit(code,layers=layers, 
                                                                    initial_stabilizers=initial_stabilizers,
                                                                    final_measurements=final_measurements,
                                                                    logical_observables=logical_observables)
                
                M = pre_circuit.number_of_instructions(["MZ", "MX", "MY"])
                Q = pre_circuit.number_of_instructions(
                    ["PAULI_CHANNEL_1"]
                )
                print(f"M, Q = {(M, Q)}")
                for p in ps:
                    if bias == math.inf:
                        m = (p * (M + Q)) / M
                        q = 0
                    else:
                        q = (p * (M + Q)) / (bias * M + Q)
                        m = q * bias
                    print(f"p, m, q = {(p, m, q)}")

                    if code_name == "Gauge2HoneycombCode" or code_name == "Gauge2FloquetColourCode":
                        code = constructor(distance,2)
                    elif code_name == "Gauge3HoneycombCode" or code_name == "Gauge3FloquetColourCode":
                        code = constructor(distance,3)
                    else:
                        code = constructor(distance)

                    stim_circuit = load_or_create_stim_circuit(
                        m,
                        q,
                        syndrome_extractor,
                        code=code,
                        layers=layers,
                        )

                    tasks[bias].append(
                    sinter.Task(
                        circuit=stim_circuit,                 
                        detector_error_model=stim_circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True, ignore_decomposition_failures=True),
                        json_metadata={
                            "code": code_name,
                            "distance": distance,
                            "bias": bias,
                            "p": round(p, 6),
                            "q": round(q,6),
                            "m": round(m,6),
                            "layers": layers,
                            "bias_type": bias_type
                        },
                    )
                    )
                    
            elif bias_type == "depolarizing_vs_dephasing":
                for p in ps:
                    p_y = p * bias / (1 + bias)
                    p_x = (p - p_y) /2
                    p_z = p_x
                    q = p_y
                    m = p_x + p_z
                    stim_circuit = load_or_create_stim_circuit_data_qubit_noise(
                        p_x,
                        p_y,
                        p_z,
                        code=constructor(distance),
                        layers=layers,
                    )

                    tasks[bias].append(
                        sinter.Task(
                            circuit=stim_circuit,                 
                            detector_error_model=stim_circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True, ignore_decomposition_failures=True),
                            json_metadata={
                                "code": code_name,
                                "distance": distance,
                                "bias": bias,
                                "p": round(p, 6),
                                "q": round(q,6),
                                "m": round(m,6),
                                "layers": layers,
                                "bias_type": bias_type
                            },
                        )
                    )

            elif bias_type == "depolarizing_vs_y":
                for p in ps:
                    p_y = p * bias / (1 + bias)
                    p_x = (p - p_y) /2
                    p_z = p_x
                    q = p_z
                    m = p_x + p_y
                    stim_circuit = load_or_create_stim_circuit_data_qubit_noise(
                        p_x,
                        p_y,
                        p_z,
                        code=constructor(distance),
                        layers=layers,
                    )

                    tasks[bias].append(
                        sinter.Task(
                            circuit=stim_circuit,                 
                            detector_error_model=stim_circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True, ignore_decomposition_failures=True),
                            json_metadata={
                                "code": code_name,
                                "distance": distance,
                                "bias": bias,
                                "p": round(p, 6),
                                "q": round(q,6),
                                "m": round(m,6),
                                "layers": layers,
                                "bias_type": bias_type
                            },
                        )
                    )
                        
    return tasks


def main(code_name, per, bias, bias_type, distances, max_n_shots, max_n_errors, decoders):
    if code_name == "HoneycombCode":
        # Collect the samples (takes a few minutes).
        code_constructor = HoneycombCode
    elif code_name == "Gauge2HoneycombCode" or code_name == "Gauge3HoneycombCode":
        code_constructor = GaugeHoneycombCode 
    elif code_name == "Gauge2FloquetColourCode" or code_name == "Gauge3FloquetColourCode":
        code_constructor = GaugeFloquetColourCode
    elif code_name == "FloquetColourCode":
        code_constructor = FloquetColourCode

    bias_tasks = get_bias_tasks(code_constructor, code_name, per, bias, bias_type, distances)
    for bias, tasks in bias_tasks.items():

        samples = sinter.collect(
            tasks=tasks,
            hint_num_tasks=len(tasks),
            num_workers=8,
            max_shots=max_n_shots,
            max_errors=max_n_errors,
            decoders=decoders,
            custom_decoders={'beliefmatching': BeliefMatchingSinterDecoder()},
            print_progress=True,
            save_resume_filepath=f'./resume_15_3/data_{code_name}.csv',
        )

def load_or_create_stim_circuit_data_qubit_noise(px,py,pz, code, layers):
  
    #    filepath = '../stim_circuits/'
    #    filepath = output_path() / f"stim_circuits/{hashed}.stim"
    filepath = Path(f"./stim_circuits/px_{px}_py_{py}_pz_{pz}_code_{type(code).__name__}_distance_{code.distance}_layers_{layers}.stim")
    if filepath.is_file():
        stim_circuit = stim.Circuit.from_file(filepath)
    else:

        compiler = AncillaPerCheckCompiler(CodeCapacityNoise(px,py,pz), CxCyCzExtractor())
        data_qubits = code.data_qubits.values()
        final_measurements = [
                Pauli(qubit, PauliLetter("X")) for qubit in data_qubits
            ]
        logical_observables = [code.logical_qubits[1].x]
        initial_stabilizers = []
        for check in code.check_schedule[0]:
            initial_stabilizers.append(Stabilizer([(0,check)],0))
        stim_circuit = compiler.compile_to_stim(
            code=code,
            layers=layers,
            initial_stabilizers = initial_stabilizers,
            final_measurements=final_measurements,
            logical_observables=logical_observables,
        )

        stim_circuit.to_file(filepath)
    return stim_circuit


def load_or_create_stim_circuit(
    m,
    q,
    syndrome_extractor,
    code,
    layers,
):
    # Save time by saving these circuits locally.
    noise_params = (q, m)
  
    #    filepath = '../stim_circuits/'
    #    filepath = output_path() / f"stim_circuits/{hashed}.stim"
    filepath = Path(f"./stim_circuits/m_{m}_q_{q}_code_{type(code).__name__}_distance_{code.distance}_layers_{layers}.stim")
    if filepath.is_file():
        stim_circuit = stim.Circuit.from_file(filepath)
    else:
        noise_model = PhenomenologicalNoise(q, m)
        compiler = AncillaPerCheckCompiler(noise_model, syndrome_extractor)
        data_qubits = code.data_qubits.values()
        final_measurements = [
                Pauli(qubit, PauliLetter("X")) for qubit in data_qubits
            ]
        logical_observables = [code.logical_qubits[1].x]
        initial_stabilizers = []
        for check in code.check_schedule[0]:
            initial_stabilizers.append(Stabilizer([(0,check)],0))
        stim_circuit = compiler.compile_to_stim(
            code=code,
            layers=layers,
            initial_stabilizers = initial_stabilizers,
            final_measurements=final_measurements,
            logical_observables=logical_observables,
        )

        stim_circuit.to_file(filepath)
    return stim_circuit


def load_or_create_stim_circuit(
    m,
    q,
    syndrome_extractor,
    code,
    layers,
):
    # Save time by saving these circuits locally.
    noise_params = (q, m)
  
    #    filepath = '../stim_circuits/'
    #    filepath = output_path() / f"stim_circuits/{hashed}.stim"
    filepath = Path(f"./stim_circuits/m_{m}_q_{q}_code_{type(code).__name__}_distance_{code.distance}_layers_{layers}.stim")
    if filepath.is_file():
        stim_circuit = stim.Circuit.from_file(filepath)
    else:
        noise_model = PhenomenologicalNoise(q, m)
        compiler = AncillaPerCheckCompiler(noise_model, syndrome_extractor)
        data_qubits = code.data_qubits.values()
        final_measurements = [
                Pauli(qubit, PauliLetter("X")) for qubit in data_qubits
            ]
        logical_observables = [code.logical_qubits[1].x]
        initial_stabilizers = []
        for check in code.check_schedule[0]:
            initial_stabilizers.append(Stabilizer([(0,check)],0))
        stim_circuit = compiler.compile_to_stim(
            code=code,
            layers=layers,
            initial_stabilizers = initial_stabilizers,
            final_measurements=final_measurements,
            logical_observables=logical_observables,
        )

        stim_circuit.to_file(filepath)
    return stim_circuit


if __name__ == "__main__":
    biases = [0,0.25, 0.5, 2, 8, 32, 9999]
    code =  str(sys.argv[1])
    ps = np.linspace(0.003, 0.02, 21)
    distances = [4,8,12,16]
    max_n_shots = 100_000
    max_n_errors = 1000
    main(
        code,
        ps,
        biases,
        "measurement_vs_data_qubit",
        distances,
        max_n_shots,
        max_n_errors,
        ["pymatching"],
    )
 