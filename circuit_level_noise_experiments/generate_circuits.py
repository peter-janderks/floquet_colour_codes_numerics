#!/usr/bin/env python3

import argparse
from typing import Callable
import numpy as np
from scipy.optimize import curve_fit
import itertools


from main.codes.tic_tac_toe.gauge.GaugeHoneycombCode import GaugeHoneycombCode
from main.codes.tic_tac_toe.gauge.GaugeFloquetColourCode import GaugeFloquetColourCode
from main.codes.tic_tac_toe.gauge.GaugeTicTacToeCode import GaugeTicTacToeCode
from main.codes.tic_tac_toe.TicTacToeCode import TicTacToeCode
from main.compiling.compilers import AncillaPerCheckCompiler, NativePauliProductMeasurementsCompiler
from main.compiling.noise.models import (
    EM3, SI1000, StandardDepolarizingNoise

)
from main.compiling.syndrome_extraction.extractors import (
    CxCyCzExtractor, NativePauliProductMeasurementsExtractor
)
from main.building_blocks.detectors.Stabilizer import Stabilizer
import pathlib


class ConstructCircuits():
    """
    Constructs 4 circuits. Two memory experiments and two stability experiments.
    """

    def __init__(self,
                 code_name,
                 per,
                 noise_model,
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
        self.noise_model = noise_model

        self.init_compiler_settings()

        self.metadata = {
            "code_name": code_name,
            "per": per,
            "noise_model": noise_model,
            "distance": distance,
            "gf_0": gf_0,
            "gf_1": gf_1,
            "gf_2": gf_2,
            "logical_observable": logical_observable,
            "n_rounds": self.rounds
        }
        meta_str = ','.join(f'{k}={v}' for k, v in self.metadata.items())

        circuit_path = out_dir / f'{meta_str}.stim'

        if circuit_path.exists():
            # the circuit already exists, no need to generate it again
            return

        stim_circuit = self.construct_circuit()

        error_distance = len(stim_circuit.detector_error_model(
            approximate_disjoint_errors=True).shortest_graphlike_error())

        if self.noise_model == "EM3":
            target_distance = distance/2
        else:
            target_distance = distance

        if error_distance < target_distance:
            raise ValueError(f"""Error in constructing circuit, the distance of the shortest graphlike error is
                             {error_distance} but it should have been {distance}. The circuit you tried to construct has metadata: {self.metadata}""")
        else:
            with open(circuit_path, 'w') as f:
                f.write(str(stim_circuit
                            ))

    def init_compiler_settings(self):

        if code_name == "GaugeHoneycombCode":
            self.code = GaugeHoneycombCode(self.distance, [
                self.gf_0, self.gf_1, self.gf_2])
        elif code_name == "GaugeFloquetColourCode":
            self.code = GaugeFloquetColourCode(self.distance, [
                self.gf_0, self.gf_1])

        if self.logical_observable == "memory_x" or self.logical_observable == "memory_z":
            self.rounds, d_x, d_z = self.code.get_number_of_rounds_for_timelike_distance(
                self.distance, graphlike=True, noise_model="circuit_level_noise")
            assert d_x == self.distance or d_z == self.distance
            self.init_compiler_settings_memory_experiment()

        elif self.logical_observable == "stability_x":
            self.rounds = self.code.get_number_of_rounds_for_single_timelike_distance(
                self.distance, 'X', graphlike=True, noise_model="circuit_level_noise")
            self.init_compiler_settings_stability()
        elif self.logical_observable == "stability_z":
            self.rounds = self.code.get_number_of_rounds_for_single_timelike_distance(
                self.distance, 'Z', graphlike=True, noise_model="circuit_level_noise")
            self.init_compiler_settings_stability()
        if self.noise_model == "EM3":
            self.syndrome_extractor = NativePauliProductMeasurementsExtractor()
        else:
            self.syndrome_extractor = CxCyCzExtractor()

    def init_compiler_settings_memory_experiment(self):

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
        self.compiler_settings = {"total_rounds": self.rounds, "initial_stabilizers": initial_stabilizers,
                                  "observables": logical_observables}

        return initial_stabilizers

    def init_compiler_settings_stability(self):
        if self.logical_observable == "stability_x":
            logical_observables = [self.code.x_stability_operator]
            if self.code_name == "GaugeHoneycombCode":
                initial_stabilizers = [Stabilizer([(0, check)], 0)
                                       for check in self.code.check_schedule[self.code.gauge_factors[0] + self.code.gauge_factors[1]]]
            elif self.code_name == "GaugeFloquetColourCode":
                initial_stabilizers = [Stabilizer([(0, check)], 0)
                                       for check in self.code.check_schedule[self.code.gauge_factors[0]]]

            final_measurements = self.code.get_possible_final_measurement(
                self.code.logical_qubits[1].z, self.rounds)

        elif self.logical_observable == "stability_z":
            logical_observables = [self.code.z_stability_operator]
            initial_stabilizers = [Stabilizer([(0, check)], 0)
                                   for check in self.code.check_schedule[0]]
            final_measurements = self.code.get_possible_final_measurement(
                self.code.logical_qubits[1].x, self.rounds)

        self.compiler_settings = {"total_rounds": self.rounds, "initial_stabilizers": initial_stabilizers,
                                  "observables": logical_observables, "final_measurements": final_measurements}

    def construct_circuit(self):
        if self.noise_model == "standard_depolarizing_noise":
            noise_model = StandardDepolarizingNoise(self.per)
        elif self.noise_model == "SI1000":
            noise_model = SI1000(self.per)

        if self.noise_model == "EM3":
            noise_model = EM3(self.per)
            compiler = NativePauliProductMeasurementsCompiler(
                noise_model, self.syndrome_extractor)
        else:
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
    parser.add_argument('--noise_model', type=str, nargs='+', required=True)
    parser.add_argument('--distance', type=int, nargs='+', required=True)
    parser.add_argument('--gf_1', type=int, nargs='+', default=0)
    parser.add_argument('--gf_2', type=int, nargs='+', default=0)
    parser.add_argument('--gf_3', type=int, nargs='+', default=[0])
    parser.add_argument('--logical_observable', nargs='+',
                        type=str, required=True)
    parser.add_argument('--out_dir', type=str, nargs='+',
                        default='out/circuits')
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    for (code_name, per, noise_model, distance, gf_1, gf_2, gf_3, logical_observable) in itertools.product(args.code_name, args.per, args.noise_model, args.distance, args.gf_1, args.gf_2, args.gf_3, args.logical_observable):

        ConstructCircuits(
            code_name,
            per,
            noise_model,
            distance,
            gf_1,
            gf_2,
            gf_3,
            out_dir,
            logical_observable
        )
