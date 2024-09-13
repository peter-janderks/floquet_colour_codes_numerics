#!/usr/bin/env python3

import argparse
import hashlib
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable
import numpy as np
import stim
from scipy.optimize import curve_fit
import itertools
import sinter
from matplotlib import pyplot as plt

from main.building_blocks.pauli import Pauli
from main.building_blocks.pauli.PauliLetter import PauliLetter
from main.codes.tic_tac_toe.FloquetColourCode import FloquetColourCode
from main.codes.tic_tac_toe.HoneycombCode import HoneycombCode
from main.codes.tic_tac_toe.gauge.GaugeHoneycombCode import GaugeHoneycombCode
from main.codes.tic_tac_toe.gauge.GaugeFloquetColourCode import GaugeFloquetColourCode
from main.codes.tic_tac_toe.TicTacToeCode import TicTacToeCode
from main.compiling.compilers.AncillaPerCheckCompiler import AncillaPerCheckCompiler
from main.compiling.noise.models import (
    PhenomenologicalNoise,

)
from main.compiling.noise.noises import OneQubitNoise
from main.compiling.syndrome_extraction.extractors.ancilla_per_check.mixed.CxCyCzExtractor import (
    CxCyCzExtractor,
)
from main.building_blocks.detectors.Stabilizer import Stabilizer
import pathlib

# src_path = pathlib.Path(__file__).parent.parent / 'src'
# assert src_path.exists()
# sys.path.append(str(src_path))


class ConstructCircuits():
    """
    Constructs 4 circuits. Two memory experiments and two stability experiments. #TODO: program stability
    """

    def __init__(self,
                 code_name,
                 per,
                 px,
                 py,
                 pz,
                 pm,
                 distance,
                 gf_0,
                 gf_1,
                 gf_2,
                 out_dir,
                 logical_observable
                 ):
        self.code_name = code_name
        self.distance = distance
        self.out_dir = out_dir
        self.per = per
        self.gf_0 = gf_0
        self.gf_1 = gf_1
        self.gf_2 = gf_2
        self.logical_observable = logical_observable
        print(self.logical_observable)
        metadata = {
            "code_name": code_name,
            "per": per,
            "px": px,
            "py": py,
            "pz": pz,
            "pm": pm,
            "distance": distance,
            "gf_0": gf_0,
            "gf_1": gf_1,
            "gf_2": gf_2,
            "logical_observable": logical_observable
        }
        meta_str = ','.join(f'{k}={v}' for k, v in metadata.items())

        circuit_path = out_dir / f'{meta_str}.stim'

        if circuit_path.exists():
            return

        self.init_compiler_settings()
        self.calculate_renomarlized_noise_model(px, py, pz, pm)
        stim_circuit = self.construct_circuit()

        error_distance = len(stim_circuit.detector_error_model(
        ).shortest_graphlike_error())
        if error_distance != distance:
            print(stim_circuit)
            print(metadata)  # , self.compiler_settings)
            print(stim_circuit.detector_error_model(
                approximate_disjoint_errors=True).shortest_graphlike_error())
            print(len(stim_circuit.detector_error_model(
                approximate_disjoint_errors=True).shortest_graphlike_error()))
            print(distance)
            print('error')
            raise ValueError('Error in constructing circuit')
        else:
            with open(circuit_path, 'w') as f:
                f.write(str(stim_circuit
                            ))

    def get_number_of_rounds_gauge_honeycomb(self):

        frequencies_sorted = sorted([self.gf_0, self.gf_1, self.gf_2])
        frequency_of_measurement_errors = sum(
            frequencies_sorted)/frequencies_sorted[0]

        frequency_of_data_qubit_errors = 2 * \
            (self.gf_0/3+self.gf_1/3+self.gf_2/3)

        return (math.ceil(max(frequency_of_measurement_errors,
                              frequency_of_data_qubit_errors)*self.distance))

    def get_number_of_rounds_gauge_floquet_colour_code(self):
        frequency_of_data_qubit_errors = self.gf_0 + self.gf_1

        frequencies_sorted = sorted([self.gf_0, self.gf_1])
        frequency_measurement_errors = 2*sum(
            frequencies_sorted)/frequencies_sorted[0]

        return max(frequency_of_data_qubit_errors, frequency_measurement_errors) * self.distance

    def init_compiler_settings(self):

        code_name_to_constructor = {
            "GaugeHoneycombCode": GaugeHoneycombCode,
            "GaugeFloquetColourCode": GaugeFloquetColourCode,
        }

        self.constructor: Callable[[int],
                                   TicTacToeCode] = code_name_to_constructor[self.code_name]
        if self.code_name == "GaugeHoneycombCode":
            self.code = self.constructor(
                self.distance, [self.gf_0, self.gf_1, self.gf_2])
            rounds = self.get_number_of_rounds_gauge_honeycomb()
        elif self.code_name == "GaugeFloquetColourCode":
            self.code = self.constructor(self.distance, [self.gf_0, self.gf_1])
            rounds = self.get_number_of_rounds_gauge_floquet_colour_code()

        initial_stabilizers = []
        if self.logical_observable == "memory_x":
            logical_observables = [self.code.logical_qubits[1].x]
            for check in self.code.check_schedule[0]:
                initial_stabilizers.append(Stabilizer([(0, check)], 0))
        elif self.logical_observable == "memory_z":
            logical_observables = [self.code.logical_qubits[0].z]
            if self.code_name == "GaugeHoneycombCode":
                for check in self.code.check_schedule[self.gf_0+self.gf_1]:
                    initial_stabilizers.append(Stabilizer([(0, check)], 0))
            elif self.code_name == "GaugeFloquetColourCode":
                for check in self.code.check_schedule[self.gf_0]:
                    initial_stabilizers.append(Stabilizer([(0, check)], 0)
                                               )

        self.compiler_settings = {"total_rounds": rounds, "initial_stabilizers": initial_stabilizers,
                                  "observables": logical_observables}

        self.syndrome_extractor = CxCyCzExtractor()

    def calculate_renomarlized_noise_model(self, px, py, pz, pm):

        pre_compiler = AncillaPerCheckCompiler(
            noise_model=PhenomenologicalNoise(1, 1),
            syndrome_extractor=self.syndrome_extractor)

        pre_circuit: stim.Circuit = pre_compiler.compile_to_circuit(
            self.code,
            **self.compiler_settings
        )

        n_measurement_error_locations = pre_circuit.number_of_instructions([
            "MZ", "MX", "MY"])
        n_data_qubit_error_locations = pre_circuit.number_of_instructions([
            "PAULI_CHANNEL_1"])
        normalized_per = self.per * (3*n_data_qubit_error_locations + n_measurement_error_locations) / (px * n_data_qubit_error_locations + py*n_data_qubit_error_locations +
                                                                                                        pz*n_data_qubit_error_locations + pm*n_measurement_error_locations)
        self.px = px * normalized_per
        self.py = py * normalized_per
        self.pz = pz * normalized_per
        self.pm = pm * normalized_per

    def construct_circuit(self):
        noise_model = PhenomenologicalNoise(
            OneQubitNoise(self.px, self.py, self.pz), self.pm)
        compiler = AncillaPerCheckCompiler(
            noise_model, self.syndrome_extractor)

        stim_circuit = compiler.compile_to_stim(
            self.code,
            **self.compiler_settings
        )
        return stim_circuit


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--code_name', nargs='+', type=str, required=True)
    parser.add_argument('--per', type=float, nargs='+', required=True)
    parser.add_argument('--px', type=float, nargs='+', required=True)
    parser.add_argument('--py', type=float, nargs='+', required=True)
    parser.add_argument('--pz', type=float, nargs='+', required=True)
    parser.add_argument('--pm', type=float, nargs='+', required=True)
    parser.add_argument('--distance', type=int, nargs='+', required=True)
    parser.add_argument('--gf_1', type=int, nargs='+', default=0)
    parser.add_argument('--gf_2', type=int, nargs='+', default=0)
    parser.add_argument('--gf_3', type=int, nargs='+', default=[0])
    parser.add_argument('--logical_observable', nargs='+',
                        type=str, required=True)
    parser.add_argument('--out_dir', type=str, nargs='+',
                        default='out/circuits')
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for (code_name, per, px, py, pz, pm, distance, gf_1, gf_2, gf_3, logical_observable) in itertools.product(args.code_name, args.per, args.px, args.py, args.pz, args.pm, args.distance, args.gf_1, args.gf_2, args.gf_3, args.logical_observable):

        print(code_name, per, px, py, pz, pm, distance,
              gf_1, gf_2, gf_3, logical_observable)
        ConstructCircuits(
            code_name,
            per,
            px,
            py,
            pz,
            pm,
            distance,
            gf_1,
            gf_2,
            gf_3,
            out_dir,
            logical_observable
        )
