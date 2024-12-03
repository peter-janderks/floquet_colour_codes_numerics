import pathlib
from generate_circuits import ConstructCircuits

dir = pathlib.Path("test")
dir.mkdir(parents=True, exist_ok=True)
constructor = ConstructCircuits(
    "GaugeFloquetColourCode",
    0.1,
    1,
    1,
    1,
    1,
    12,
    4,
    4,
    1,
    1,
    0,
    out_dir=dir,
    logical_observable="memory_x",
)

constructor = ConstructCircuits(
    "GaugeFloquetColourCode",
    0.1,
    1,
    1,
    1,
    1,
    16,
    4,
    4,
    1,
    1,
    0,
    out_dir=dir,
    logical_observable="memory_x",
)
