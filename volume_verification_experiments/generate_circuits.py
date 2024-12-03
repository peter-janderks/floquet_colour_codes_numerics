#!/usr/bin/env python3

import argparse
from typing import Callable
import stim
import itertools

from main.codes.tic_tac_toe.gauge.GaugeHoneycombCode import GaugeHoneycombCode
from main.codes.tic_tac_toe.gauge.GaugeFloquetColourCode import GaugeFloquetColourCode
from main.codes.tic_tac_toe.gauge.GaugeTicTacToeCode import GaugeTicTacToeCode
from main.codes.tic_tac_toe.TicTacToeCode import TicTacToeCode
from main.compiling.compilers.AncillaPerCheckCompiler import AncillaPerCheckCompiler
from main.compiling.noise.models import (
    PhenomenologicalNoise,
    StandardDepolarizingNoise

)
from main.compiling.noise.noises import OneQubitNoise
from main.compiling.syndrome_extraction.extractors.ancilla_per_check.mixed.CxCyCzExtractor import (
    CxCyCzExtractor,
)
from main.building_blocks.detectors.Stabilizer import Stabilizer
import pathlib


class ConstructCircuits():
    """
    Constructs 4 circuits. Two memory experiments and two stability experiments. #TODO: program stability
    """

    def __init__(self,
                 code_name,
                 noise_model,
                 d_x,
                 d_z,
                 h,
                 gf_0,
                 gf_1,
                 gf_2,
                 out_dir,
                 logical_observable
                 ):
        self.code_name = code_name
        self.noise_model_name = noise_model
        self.d_x = d_x
        self.d_z = d_z
        self.h = h
        self.out_dir = out_dir
        self.gf_0 = gf_0
        self.gf_1 = gf_1
        self.gf_2 = gf_2

        self.logical_observable = logical_observable
        metadata = {
            "code_name": code_name,
            "noise_model": noise_model,
            "d_x": d_x,
            "d_z": d_z,
            "h": h,
            "gf_0": gf_0,
            "gf_1": gf_1,
            "gf_2": gf_2,
            "logical_observable": logical_observable
        }
        meta_str = ','.join(f'{k}={v}' for k, v in metadata.items())

        circuit_path = out_dir / f'{meta_str}.stim'

        if circuit_path.exists():
            # the circuit already exists, no need to generate it again
            return

        self.init_compiler_settings()
        stim_circuit = self.construct_circuit()

        error_distance = len(stim_circuit.detector_error_model(approximate_disjoint_errors=True
                                                               ).shortest_graphlike_error())
        if error_distance < d_z:
            raise ValueError(f"""Error in constructing circuit, the distance of the shortest graphlike error is
                             {error_distance} but it should have been {d_x}. The circuit you tried to construct has metadata: {metadata}""")
        else:
            with open(circuit_path, 'w') as f:
                f.write(str(stim_circuit
                            ))

    def init_compiler_settings(self):

        code_name_to_constructor = {
            "GaugeHoneycombCode": GaugeHoneycombCode,
            "GaugeFloquetColourCode": GaugeFloquetColourCode,
        }
        self.constructor: Callable[[int],
                                   TicTacToeCode] = code_name_to_constructor[self.code_name]

        if self.noise_model_name == "phenomenological":
            self.noise_model = PhenomenologicalNoise(0.001)
        elif self.noise_model_name == "circuit_level_noise":
            self.noise_model = StandardDepolarizingNoise(0.0005)
        print(self.gf_0, self.gf_1, self.gf_2)
        self.code: GaugeTicTacToeCode = self.constructor(
            [self.d_x, self.d_z], [self.gf_0, self.gf_1, self.gf_2])

        rounds, h_x, h_z = self.code.get_number_of_rounds_for_timelike_distance(
            self.h, graphlike=True)
        assert h_x == self.h or h_z == self.h
        if self.logical_observable == "memory_x" or self.logical_observable == "memory_z":
            self.init_compiler_settings_memory_experiment(rounds)

        elif self.logical_observable == "stability_x" or self.logical_observable == "stability_z":
            self.init_compiler_settings_stability(rounds)

        self.syndrome_extractor = CxCyCzExtractor()

    def init_compiler_settings_memory_experiment(self, rounds):

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

        return initial_stabilizers

    def init_compiler_settings_stability(self, rounds):
        if self.logical_observable == "stability_x":
            logical_observables = [self.code.x_stability_operator]
            if self.code_name == "GaugeHoneycombCode":
                initial_stabilizers = [Stabilizer([(0, check)], 0)
                                       for check in self.code.check_schedule[self.code.gauge_factors[0] + self.code.gauge_factors[1]]]
            elif self.code_name == "GaugeFloquetColourCode":
                initial_stabilizers = [Stabilizer([(0, check)], 0)
                                       for check in self.code.check_schedule[self.code.gauge_factors[0]]]

            final_measurements = self.code.get_possible_final_measurement(
                self.code.logical_qubits[1].z, rounds)

        elif self.logical_observable == "stability_z":
            logical_observables = [self.code.z_stability_operator]
            initial_stabilizers = [Stabilizer([(0, check)], 0)
                                   for check in self.code.check_schedule[0]]
            final_measurements = self.code.get_possible_final_measurement(
                self.code.logical_qubits[1].x, rounds)

        self.compiler_settings = {"total_rounds": rounds, "initial_stabilizers": initial_stabilizers,
                                  "observables": logical_observables, "final_measurements": final_measurements}

    def construct_circuit(self):
        compiler = AncillaPerCheckCompiler(
            self.noise_model, self.syndrome_extractor)

        stim_circuit = compiler.compile_to_stim(
            self.code,
            **self.compiler_settings
        )
        return stim_circuit


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--code_name', type=str, required=True)
    parser.add_argument('--noise_model', type=str, required=True)
    parser.add_argument('--d_x', type=int, required=True)
    parser.add_argument('--d_z', type=int, required=True)
    parser.add_argument('--h', type=int, required=True)
    parser.add_argument('--gf_0', type=int, default=0)
    parser.add_argument('--gf_1', type=int, default=0)
    parser.add_argument('--gf_2', type=int, default=0)
    parser.add_argument('--logical_observable', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='out/circuits')
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ConstructCircuits(
        args.code_name,
        args.noise_model,
        args.d_x,
        args.d_z,
        args.h,
        args.gf_0,
        args.gf_1,
        args.gf_2,
        out_dir,
        args.logical_observable
    )
