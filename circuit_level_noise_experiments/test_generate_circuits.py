import pathlib
from generate_circuits import ConstructCircuits


def test_hcc_SD6():
    out_dir = pathlib.Path('out/test_output/')
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    ConstructCircuits(
        code_name='GaugeFloquetColourCode',
        per=0.1,
        noise_model='SI1000',
        distance=4,
        gf_0=3,
        gf_1=1,
        gf_2=0,
        out_dir=out_dir,
        logical_observable='memory_x'
    )


test_hcc_SD6()


def test_hcc_SD6():
    out_dir = pathlib.Path('out/test_output/')
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    ConstructCircuits(
        code_name='GaugeFloquetColourCode',
        per=0.1,
        noise_model='standard_depolarizing_noise',
        distance=4,
        gf_0=1,
        gf_1=3,
        gf_2=0,
        out_dir=out_dir,
        logical_observable='stability_x'
    )


test_hcc_SD6()
