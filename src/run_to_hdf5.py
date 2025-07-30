import numpy as np
import dpdata 
import h5py
import re

from ase import Atoms
from ase.io import Trajectory
from dpdata.unit import LengthConversion, EnergyConversion, ForceConversion
from run_calculator import SetCalculator


def ObtainHLData(args):
    """ Obtain and save energies and forces from initial XYZ file.
    
    Parameters:
    -----------
        args (Namespace):
            Parsed command line arguments containing the initial XYZ file and output HDF5 file.
        
        
    Returns:
    --------
        None
    
    """
    dih_ints = np.arange(36)
    dihedrals, energies, coords, forces = [], [], [], []
    for dih in dih_ints:
        try:
            elements, positions = Load_Positions(f"pos_{dih}_{args.initxyz}.dat")
            energies.append(Load_Energy(f"ener_{dih}_{args.initxyz}.dat"))
            coords.append(positions)
            forces.append(Load_Forces(f"forces_{dih}_{args.initxyz}.dat"))
        except:
            print(f"Skipping dihedral {dih} due to missing files.")
            continue
        dihedrals.append(dih*10.0)

    #hl_data = Setup_Output_for_DeePMD(energies, coords, forces, elements)
    #print("HL Output data prepared for DeePMD:", hl_data)

    #profile_name
    match = re.search(r'(\d+-\d+-\d+-\d+)\.xyz', args.initxyz)
    profile_name = match.group(1) if match else "profile"
    Write_Dihedral_Profile(dihedrals, energies, f"spice_{profile_name}.dat")
    
    return

def Write_Dihedral_Profile(dihedrals, energies, filename):
    """ Write dihedral profile to a file.
    
    Parameters:
    -----------
        dihedrals (list): 
            List of dihedral angles.
        energies (list): 
            List of energies corresponding to the dihedral angles.
        filename (str): 
            Name of the output file to write the dihedral profile.
    
    """
    ev_to_kcal = 23.0605  # Conversion factor from eV to kcal/mol
    with open(filename, 'w') as f:
        for dih, energy in zip(dihedrals, energies):
            f.write(f"{dih/10} {energy}\n")

def Obtain_XTB_LL(args, hl_data):
    """ Take HL structures, and then run single point calculations on them."""
    calculator = SetCalculator(
        calculator_name="xtb",
        charge=args.charge,
        method= "GFN2-xTB"
    )

    energies, coords, forces = [], [], []
    for coords in hl_data['coordinates']:
        atoms = Position_To_Molecule(hl_data['elements'], coords)
        atoms.calc = calculator
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())

    ll_data = Setup_Output_for_DeePMD(energies, hl_data['coordinates'], forces, hl_data['elements'])
    return ll_data

def write_final_hdf5(args, hl_data, ll_data):
    """ Subtract the HL data from the LL data and write to HDF5."""
    diff_data = {
        "energies": np.array(hl_data['energies']) - np.array(ll_data['energies']),
        "coordinates": np.array(ll_data['coordinates']),
        "forces": np.array(hl_data['forces']) - np.array(ll_data['forces']),
        "elements": np.array(ll_data['elements'])
    }
    types = np.unique(ll_data['elements'])
    

def to_hdf5(elements, coordinates, energies, forces, filename):
    """ Save data to HDF5 file."""

    type_map_raw = np.unique(elements)
    type_map = []
    for elem in elements:
        if elem not in type_map_raw:
            raise ValueError(f"Element {elem} not found in type map.")
        type_map.append(np.where(type_map_raw == elem)[0][0])
        
    type_map_raw_fixed = np.array(type_map_raw, dtype='S')
    
    coordinates = np.array(coordinates).reshape(len(coordinates), -1)
    forces = np.array(forces).reshape(len(forces), -1)
    

    energies = np.array(energies)
    
    with h5py.File(filename, 'w') as hdf5_file:
        hdf5_file.create_dataset('nopbc', data=True) # No periodic boundary conditions
        grp = hdf5_file.create_group('set.000')
        grp.create_dataset('coord.npy', data=coordinates)
        grp.create_dataset('energy.npy', data=energies)
        grp.create_dataset('force.npy', data=forces)
        hdf5_file.create_dataset('type.raw', data=type_map)
        hdf5_file.create_dataset('type_map.raw', data=type_map_raw_fixed)
    # No need to explicitly close the file or delete it, as it's handled by the context manager


    

def Position_To_Molecule(elements, positions):
    """ Convert positions to ASE Atoms object."""
    atoms = Atoms(symbols=elements, positions=positions)
    return atoms
                    
def Load_Energy(filename):
    """ Load energies from a file.
    
    Parameters:
    -----------
        filename (str): 
            Path to the file containing energies.
    
    Returns:
    --------
        float: 
            Energy read from the file.
    
    """
    with open(filename, "r") as f:
        line = f.readline().strip().split()
        energy = float(line[1])
    return energy

def Load_Positions(filename):
    """ Load positions from a file.
    
    Parameters:
    -----------
        filename (str): 
            Path to the file containing positions.
    
    Returns:
    --------
        list: 
            List of positions read from the file.
    
    """
    positions = []
    elements = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                pos = [float(coord) for coord in parts[1:4]]
                positions.append(pos)
                elements.append(parts[0])
    return elements, positions

def Load_Forces(filename):
    """ Load forces from a file.
    
    Parameters:
    -----------
        filename (str): 
            Path to the file containing forces.
    
    Returns:
    --------
        list: 
            List of forces read from the file.
    
    """
    forces = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                force = [float(coord) for coord in parts[1:4]]
                forces.append(force)
    return forces

def Setup_Output_for_DeePMD(energies, coordinates, forces, elements):
    """ This takes in numpy arrays of energies, coordinates, and forces, and returns a dictionary."""
    # dpdata calculators take in arguments unitA, unitB, and outputs what you need to multiply unitB by to get unitA.
    length_convert = LengthConversion("bohr", "angstrom").value()
    energy_convert = EnergyConversion("hartree", "eV").value()
    force_convert = ForceConversion("hartree/bohr", "eV/angstrom").value()

    hl_data = {
        "energies": np.array(energies) * energy_convert,
        "coordinates": np.array(coordinates) * length_convert,
        "forces": np.array(forces) * force_convert,
        "elements": np.array(elements)
    }



    return hl_data



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run save energies and forces to HDF5 file.")
    parser.add_argument("--initxyz", type=str, required=True, help="Initial XYZ, which generated the INITIAL files.")
    parser.add_argument("--hdf5", type=str, default="out.hdf5", required=True, help="Output HDF5 file to save energies and forces.")
    parser.add_argument("--charge", type=int, default=0, help="Charge of the system for the calculator.")
    

    args = parser.parse_args()

    hl_data = ObtainHLData(args)
    #ll_data = Obtain_XTB_LL(args, hl_data=hl_data)

