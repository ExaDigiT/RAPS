"""Shared constants for workloads."""

JOB_NAMES = [
    "LAMMPS", "GROMACS", "VASP", "Quantum ESPRESSO", "NAMD",
    "OpenFOAM", "WRF", "AMBER", "CP2K", "nek5000", "CHARMM",
    "ABINIT", "Cactus", "Charm++", "NWChem", "STAR-CCM+",
    "Gaussian", "ANSYS", "COMSOL", "PLUMED", "nekrs",
    "TensorFlow", "PyTorch", "BLAST", "Spark", "GAMESS",
    "ORCA", "Simulink", "MOOSE", "ELK"
]

ACCT_NAMES = [f"ACT{i:02d}" for i in range(1, 15)]
MAX_PRIORITY = 500000
