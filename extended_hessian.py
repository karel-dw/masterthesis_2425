from pathlib import Path
import numpy as np
from ase.units import kB
from psiflow.geometry import Geometry
from psiflow.functions import EnergyFunction, MACEFunction, PlumedFunction
from ase.units import bar, kB

from transform import (
    cell_jac, 
    pos_jac, 
    transform_cell_regular_to_different, 
    transform_pos_regular_to_different, 
    transform_pos_different_to_regular, 
    transform_cell_different_to_regular,
)

PLUMED_INPUT = """UNITS LENGTH=A ENERGY=eV
vol: VOLUME
"""

def get_bias_function(pressure: float, temperature: float, number_of_atoms: int) -> PlumedFunction:
    c1 = pressure * 10 * bar  # Convert to ase units
    c2 = kB * temperature * (number_of_atoms - 2)

    plumed_str = PLUMED_INPUT
    plumed_str += '\n'
    plumed_str += f'pv: MATHEVAL ARG=vol VAR=x FUNC={c1}*x PERIODIC=NO  \n'
    plumed_str += f'lnv: MATHEVAL ARG=vol VAR=x FUNC={c2}*log(x) PERIODIC=NO  \n'   # not easy to find if log(x) calculates ln(x)
    plumed_str += f'bv: MATHEVAL ARG=pv,lnv VAR=x,y FUNC=x-y PERIODIC=NO  \n'
    plumed_str += 'BIASVALUE ARG=bv\n'
    return PlumedFunction(plumed_str)


def compute_row(
    perturbed_geometries: list[Geometry],
    functions: list[EnergyFunction],
    delta: float,
    coo,
    cell_0
):
    outputs = None
    for func in functions:
        if outputs is None:
            outputs = func.compute(perturbed_geometries)
        else:
            out = func.compute(perturbed_geometries)
            for prop in outputs:
                outputs[prop] += out[prop]
    num_atoms = outputs['forces'].shape[1]          #shape = (2,N,3)
    row = np.zeros(3 * num_atoms + 9)

    # insert first 3N elements
    # compute positional diff gradients for each perturbation couple
    pos_grad = []
    for i in range(2):
        cell_perturb = np.copy(perturbed_geometries[i].cell)
        denergy_dpos = np.copy((-1.0) * outputs['forces'][i])
        denergy_dposdiff = denergy_dpos @ pos_jac(coo,cell_perturb,cell_0,num_atoms)
        pos_grad.append(denergy_dposdiff)
    row[:3 * num_atoms] = np.copy((pos_grad[0] - pos_grad[1]) / (2 * delta)).flatten()


    # insert first 9 elements
    # compute cell diff gradients for each perturbation couple
    cell_grad = []
    for i in range(2):
        cell_perturb = np.copy(perturbed_geometries[i].cell)
        volume = np.linalg.det(cell_perturb)
        denergy_dcell = np.copy(volume * (outputs['stress'][i] @ np.linalg.inv(cell_perturb)).T)
        denergy_dcelldiff = denergy_dcell.flatten() @ cell_jac(coo,cell_0,num_atoms)
        cell_grad.append(denergy_dcelldiff)
    row[3 * num_atoms:] = np.copy((cell_grad[0] - cell_grad[1]) / (2 * delta))

    return row


def compute_hessian(
    geometry: Geometry,
    functions: list[EnergyFunction],
    d_pos_diff: float = 1e-5,
    d_cell_diff: float = 1e-5,
    coo = None
) -> np.ndarray:
    
    num_atoms = len(geometry)

    cell_0 = np.copy(geometry.cell)
    diff_cell_0 = transform_cell_regular_to_different(coo,cell_0,cell_0,num_atoms)
    if coo == 'sF' or coo == 'rF' or coo == 'dF':
        assert np.allclose(diff_cell_0, np.identity(3)), 'F_0 should be identity matrix'
    positions_0 = np.copy(geometry.per_atom.positions)
    diff_positions_0 = transform_pos_regular_to_different(coo,positions_0,cell_0,cell_0,num_atoms)
    rows = []  # list of  3N x 9 rows of extended hessian

    # perturb diff coordinates
    for i in range(num_atoms):
        for j in range(3):
            perturbed_geometries = []
            for direction in [+1, -1]:
                tmp = geometry.copy()
                diff_positions = np.copy(diff_positions_0)
                diff_positions[i, j] += direction * d_pos_diff

                # convert to positions and regular unit cell
                positions = transform_pos_different_to_regular(coo,diff_positions,cell_0,cell_0,num_atoms)
                tmp.per_atom.positions = np.copy(positions)
                perturbed_geometries.append(tmp)

            row = compute_row(perturbed_geometries, functions, d_pos_diff,coo, cell_0)
            rows.append(row)

    # perturb cell
    for i in range(3):
        for j in range(3):
            perturbed_geometries = []
            for direction in [+1, -1]:
                tmp = geometry.copy()
                diff_cell = np.copy(diff_cell_0)
                diff_cell[i, j] += direction * d_cell_diff
                cell = transform_cell_different_to_regular(coo,diff_cell,cell_0,num_atoms)

                positions = transform_pos_different_to_regular(coo,diff_positions_0,cell,cell_0,num_atoms)
                tmp.per_atom.positions = np.copy(positions)
                tmp.cell = np.copy(cell)
                perturbed_geometries.append(tmp)

            row = compute_row(perturbed_geometries, functions, d_cell_diff,coo, cell_0)
            rows.append(row)

    return np.array(rows)


if __name__ == '__main__':
    
    structure_names = ['Ice_VIII','Ice_XI']
    for structure_name in structure_names:
        pressures = [0.1] #MPa
        for pressure in pressures:
            temperatures = [150] #K
            for temperature in temperatures:
                coordinates = ['dh'] #,'dh'
                for coordinate in coordinates:
                    path_opt = Path.cwd() / 'opt_structures_biased' / str('opt_' + structure_name + '_'+str(temperature)+'K_'+str(pressure)+'MPa.xyz')
                    path_hessian = Path.cwd() / 'hessians_biased' / str('hessian_'+ coordinate + '_' + structure_name + '_'+str(temperature)+'K_'+str(pressure)+'MPa.npy')

                    opt_geo = Geometry.load(path_opt)
                    mace_function = MACEFunction(
                        model_path='ani500k_cc_cpu.model',
                        ncores=4,
                        device='cpu',
                        dtype='float64',
                        atomic_energies={},
                    )
                    plumed_function = get_bias_function(pressure=pressure, temperature=temperature, number_of_atoms=len(opt_geo))

                    if not Path(path_hessian).exists():
                        hessian = compute_hessian(opt_geo, [mace_function, plumed_function], coo=coordinate)
                        np.save(path_hessian, hessian)
                    hessian = np.load(path_hessian)

                    print('max asymmetry: {}'.format(np.max(np.abs(hessian - hessian.T))))
                    values = np.linalg.eigvalsh(hessian)   # assumes that matrix is symmetric! but function does not check it
                    print('lowest 7 eigenvalues: {}'.format(values[:7]))
                    print('condition number of nonzero modes: {}'.format(values[-1] / values[6]))