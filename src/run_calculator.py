from ase.io import read
from ase.optimize import BFGS
from ase.constraints import FixAtoms
import os, shutil, re

from argparse import ArgumentParser


def GetMolecules(input_file):
    """
    Read molecules from an XYZ file.
    
    Parameters:
    -----------
        input_file (str): Path to the input XYZ file.
    
    Returns:
    --------
        list: List of ASE Atoms objects.
    """
    return read(input_file, index=':')

def SetCalculator(calculator_name, basis_set=None, method=None, charge=0, num_threads=12):  
    """
    Set the ASE calculator based on the provided name.
    
    Parameters:
    -----------
        calculator_name (str): Name of the calculator to use.
    
    Returns:
    --------
        ase.calculators.Calculator: Configured ASE calculator.
    """
    approved_calculators = ["psi4",  "dftb", "xtb", "aimnet", "deepmd"]
    supports_multithreads = {'psi4': True, 'dftb': False, 'xtb': False, 'aimnet': False, 'deepmd': False}

    if calculator_name.lower() not in approved_calculators:
        raise ValueError(f"Calculator '{calculator_name}' is not supported. Choose from {approved_calculators}.")
    if calculator_name.lower() == "psi4":
        try:
            from ase.calculators.psi4 import Psi4
            return Psi4(method=method, basis=basis_set, scf_type='df', charge=charge, num_threads=num_threads, mem='10GB')
        except ImportError:
            raise ImportError("Psi4 calculator is not installed. Please install it to use this calculator.")
    elif calculator_name.lower() == "xtb":
        try:
            from xtb.ase.calculator import XTB as OldXTB
            return OldXTB(charge=charge, method=method)
        except ImportError:
            raise ImportError("XTB calculator is not installed. Please install it to use this calculator.")
    elif calculator_name.lower() == "dftb":
        try:
            from ase.calculators.dftb import Dftb
            return Dftb(label='dftb', Hamiltonian_SCC='Yes', skf_loc=basis_set, Hamiltonian_MaxAngularMomentum={'H': 's', 'C': 'p', 'O': 'p', 'S': 'p', 'Cl': 'p', 'F': 'p'}, charge=charge)
        except ImportError:
            raise ImportError("DFTB calculator is not installed. Please install it to use this calculator.")
    elif calculator_name.lower() == "aimnet":
        try:
            from aimnet import load_AIMNetMT_ens, AIMNetCalculator
            model_gas = load_AIMNetMT_ens().cuda()
            return AIMNetCalculator(model=model_gas)
        except ImportError:
            raise ImportError("AIMNet calculator is not installed. Please install it to use this calculator.")
    elif calculator_name.lower() == "deepmd":
        try:
            if ".pb" in method:
                from deepmd.calculator import DP
                return DP(model=method)
            elif ".model" in method:
                from mace.calculators import MACECalculator
                return MACECalculator(model_paths=method)
        except ImportError:
            raise ImportError("DeepMD calculator is not installed. Please install it to use this calculator.")
        
    
def IndividualCalc(args, calculator, molecules):
    """ Run calculations on individual structures in the input file.
    
    Parameters:
    -----------
        args (Namespace):
            Parsed command line arguments.
        calculator (ase.calculators.Calculator):
            ASE calculator to use.
        molecules (list):
            List of ASE Atoms objects.
    Returns:
    --------
        None
    
    """
    energies, forces = [], []
    max_conformations = args.max_conformations if args.max_conformations != -1 else len(molecules)

    for i, mol in enumerate(molecules[:max_conformations:args.stride]):
        mol.calc = calculator
        if args.constraints:
            indices = list(map(int, args.constraints.split(',')))
            mol.set_constraint(FixAtoms(indices=indices))

        if args.optimize:
            optimizer = BFGS(mol)
            optimizer.run(fmax=args.tol)

        energy = mol.get_potential_energy()
        forces.append(mol.get_forces())
        energies.append(energy)
        mol.write(f"{args.tag}_structure_{i * args.stride}.xyz")
        print(f"Structure {i * args.stride}: {energy} eV")

        with open(f"{args.tag}_energies.txt", "w") as f:
            for i, energy in enumerate(energies):
                f.write(f"{i * args.stride}: {energy} eV\n")
        with open(f"{args.tag}_forces.txt", "w") as f:
            for i, force in enumerate(forces):
                f.write(f"{i * args.stride}: {force.tolist()}\n")
    return

def SingleCalc(args, calculator, molecules):
    """ Run a single calculation on a specified structure.
    
    Parameters:
    -----------
        args (Namespace): 
            Parsed command line arguments.
        calculator (ase.calculators.Calculator): 
            ASE calculator to use.
        molecules (list): 
            List of ASE Atoms objects.
    
    Raises:
    -------
        ValueError: If the specified structure index is out of bounds or invalid.
    
    """
    energies, forces = [], []
    if args.structure < 0 or args.structure >= len(molecules):
        raise ValueError(f"Structure index {args.structure} is out of bounds for the number of structures ({len(molecules)}).")
    if args.structure == -1:
        raise ValueError("Please specify a valid structure index to process.")
    
    # check if the calculation has already been done
    if os.path.exists(f"ener_{args.structure}_{args.input}.dat"):
        print(f"Calculation for structure {args.structure} already done. Skipping.")
        return

    mol = molecules[args.structure]
    mol.calc = calculator
    if args.constraints:
        if args.constraints == "filename":
            match = re.search(r'_(\d+(?:-\d+)*)\.xyz$', args.input)
            if match:
                indices = list(map(int, match.group(1).split('-')))
                mol.set_constraint(FixAtoms(indices=indices))
            else:
                raise ValueError("Could not extract constraints from filename.")
        else:
            indices = list(map(int, args.constraints.split(',')))
            mol.set_constraint(FixAtoms(indices=indices))

    if args.optimize:
        optimizer = BFGS(mol)
        optimizer.run(fmax=args.tol)

    energy = mol.get_potential_energy()
    forces.append(mol.get_forces())
    positions = mol.get_positions()

    print(f"Structure {args.structure}: {energy} eV")
    with open(f"ener_{args.structure}_{args.input}.dat",'w') as f:
        f.write(f"{args.structure} {energy}\n")

    with open(f"pos_{args.structure}_{args.input}.dat", 'w') as f:
        print(f"Structure {args.structure} positions:")
        for i, pos in enumerate(positions):
            elem = mol.get_chemical_symbols()[i]
            line = f"{elem} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"
            f.write(line)
            print(line.strip())

    with open(f"forces_{args.structure}_{args.input}.dat", 'w') as f:
        print(f"Structure {args.structure} forces:")
        for i, atom_force in enumerate(forces[0]):
            elem = mol.get_chemical_symbols()[i]
            line = f"{elem} {atom_force[0]:.6f} {atom_force[1]:.6f} {atom_force[2]:.6f}\n"
            f.write(line)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run a general calculator on a set of structures.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input XYZ file")
    parser.add_argument("-t", "--tag", type=str, required=True, help="Tag for output files")
    parser.add_argument("-o", "--optimize", action="store_true", help="Whether to optimize the geometry")
    parser.add_argument("-c", "--constraints", type=str, default=None, help="Optional string of constraints")
    parser.add_argument("--tol", type=float, default=0.02, help="Optimizer tolerance (default: 0.02)")
    parser.add_argument("--calculator", type=str, required=True, help="ASE calculator to use")
    parser.add_argument("--basis_set", type=str, default=None, help="Basis set for the calculator (if applicable)")
    parser.add_argument("--method", type=str, default=None, help="Method for the calculator (if applicable)")
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge (default: 0)")
    parser.add_argument("--num_threads", type=int, default=12, help="Number of threads to use (default: 12)")
    parser.add_argument("--stride", type=int, default=1, help="Stride for processing structures (default: 1)")
    parser.add_argument("--max_conformations", type=int, default=-1, help="Maximum number of conformations to process (default: -1, meaning all conformations)")
    parser.add_argument("--structure", type=int, default=-1, help="Index of the structure to process (default: -1)")
    args = parser.parse_args()

    molecules = GetMolecules(args.input)

    calculator = SetCalculator(
        calculator_name=args.calculator,
        basis_set=args.basis_set,
        method=args.method,
        charge=args.charge,
        num_threads=args.num_threads
    )
    
    if args.structure == -1:
        IndividualCalc(args, calculator, molecules)
    else:
        SingleCalc(args, calculator, molecules)