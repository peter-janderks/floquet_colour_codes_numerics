import stim

circuit = stim.Circuit.from_file(
    "out/circuits/code_name=GaugeFloquetColourCode,per=0.001,px=1.0,py=1.0,pz=1.0,pm=1.0,distance=8,gf_0=2,gf_1=3,gf_2=0.stim")
print(circuit.num_detectors)
print(circuit.num_measurements)
print(circuit.detector_error_model().to_file("dem2.stim"))

# # print(circuit)
# circuit = stim.Circuit.from_file(
#     "out/circuits/code_name=GaugeFloquetColourCode,per=0.001,px=1.0,py=1.0,pz=1.0,pm=1.0,distance=8,gf_0=3,gf_1=2,gf_2=0.stim")
# print(circuit.num_detectors)
# print(circuit.num_measurements)
# print(circuit)
circuit = stim.Circuit.from_file(
    "out/circuits/code_name=GaugeFloquetColourCode,per=0.001,px=1.0,py=1.0,pz=1.0,pm=1.0,distance=12,gf_0=2,gf_1=3,gf_2=0.stim")
print(circuit.num_detectors)
print(circuit.num_measurements)
# print(circuit.detector_error_model().to_file("dem.stim"))
circuit = stim.Circuit.from_file(
    "out/circuits/code_name=GaugeFloquetColourCode,per=0.001,px=1.0,py=1.0,pz=1.0,pm=1.0,distance=12,gf_0=3,gf_1=2,gf_2=0.stim")
print(circuit.num_detectors)
print(circuit.num_measurements)
