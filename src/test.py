import numpy as np 
from run_to_hdf5 import to_hdf5
import dpdata
import h5py

elements = ['H', 'O']  # Example elements
coordinates = [[[0,0,0],[1,1,1]], [[1,0,0],[2,1,1]]]  # Example coordinates
energies = [0.0, -1.0]  # Example energies
forces = coordinates  # Example forces

to_hdf5(elements, coordinates, energies, forces, 'test.hdf5')

sys = dpdata.LabeledSystem('test.hdf5', fmt='deepmd/hdf5')
print(sys)