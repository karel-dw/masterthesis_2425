from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase.calculators.calculator import Calculator, all_properties
from ase.units import kB, bar
from psiflow.geometry import Geometry
from psiflow.functions import MACEFunction, PlumedFunction
import os
from psiflow.utils.apps import get_attribute

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

class FunctionCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def calculate(
        self,
        atoms=None,
        properties=all_properties,
        system_changes=None,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        geometry = Geometry.from_atoms(atoms)
        outputs = self.function([geometry])
        self.results = {name: array[0] for name, array in outputs.items()}

def generate_inverse(
    eigvals,
    eigvecs,
    n
):
    '''
    sqrt(A^(-1)) = sum_{lambda}(sqrt(lambda^(-1))*|v_{lambda}><v_{lambda}|)
    '''
    inverse = np.zeros((n,n))
    for i in range(n):
        eigval = eigvals[i]
        eigvec = eigvecs[:,i]
        if abs(eigval) > 0.01:
            proj = np.outer(eigvec,eigvec)
            inverse += (1/eigval)*proj
    return inverse

def get_geo_cell(
    samples_in_diff_coo,
    opt_geo,
    npt,
    coo
):
    '''
    npt=True: starting from a 3N+9 row in sh space around zero mean, go to geometry object with correct positions and cell
    '''
    opt_pos = np.copy(opt_geo.per_atom.positions)
    opt_cell = np.copy(opt_geo.cell)
    n = len(opt_geo)
    nms_geo = []

    if npt:
        for sample_diff in samples_in_diff_coo:
            new_cell = np.copy(opt_cell + transform_cell_different_to_regular(coo,sample_diff[3*n:].reshape((3,3)),opt_cell,n))
            opt_pos_diff = transform_pos_regular_to_different(coo,opt_pos,opt_cell,opt_cell,n)
            new_pos = transform_pos_different_to_regular(coo, opt_pos_diff + sample_diff[:3*n].reshape((n,3)),new_cell,opt_cell,n)

            new_geo = opt_geo.copy()
            new_geo.cell = new_cell
            new_geo.per_atom.positions = new_pos

            nms_geo.append(new_geo)
    else:
        for sample_diff in samples_in_diff_coo:
            new_pos = opt_pos + transform_pos_different_to_regular(coo,sample_diff.reshape((n,3)),opt_cell,opt_cell,n)

            new_geo = opt_geo.copy()
            new_geo.per_atom.positions = new_pos

            nms_geo.append(new_geo)
    return nms_geo

def generate_nms_samples_energies(
    hessian,
    opt_geo,
    npt = True,
    n_samples = 100,
    Temp = 50,
    coo = None
):
    '''
    generate normally distributed samples in sh space and rh space and compute the harmonic and mlp energy respectively
    '''
    if not npt:
        n = len(opt_geo)
        hessian = hessian[:3*n,:3*n]
    eigvals,eigvecs = np.linalg.eig(hessian)

    assert np.min(eigvals) > -0.01, 'negative eigen value'
    if np.max(np.iscomplex(eigvals)):
        print('Complex eigenvalues!!!')
        print('max asymmetry: {}'.format(np.max(np.abs(hessian - hessian.T))))
    
    hessian = np.copy((hessian + hessian.T)/2)
    eigvals,eigvecs = np.linalg.eig(hessian)
    
    sigma_squared = generate_inverse(eigvals, eigvecs, len(eigvals))

    ## Generate harmonic energies = [0.5*s1@H@s1.T, ...]
    mean = np.zeros(len(sigma_squared))     ## for e_harm, it holds that (sample -g0)*H*(sample -g0) comes down to sampling around the origin
    cov = kB*Temp*sigma_squared
    rng = np.random.default_rng()
    samples_in_diff_coo = rng.multivariate_normal(mean,cov, size=n_samples)
    energy_harm = []
    for s in samples_in_diff_coo:
        energy_harm.append(0.5*s@hessian@s.T)
    energy_opt_bias = get_attribute(opt_geo, "energy")
    energy_harm = np.asarray(energy_harm) + energy_opt_bias

    samples_in_r_h = get_geo_cell(samples_in_diff_coo, opt_geo, npt, coo)
    
    ## Generate mlp energies = mlp([geo1,...])
    outputs = mace_function.compute(samples_in_r_h)
    energy_mlp = outputs['energy']

    ## Generate bias energies = bias([geo1,...])
    outputs = plumed_function.compute(samples_in_r_h)
    energy_bias = outputs['energy']

    return (samples_in_diff_coo,samples_in_r_h,np.asarray(energy_harm), np.asarray(energy_mlp),np.asarray(energy_bias))


if __name__ == '__main__':
    mace_function = MACEFunction(
        model_path='ani500k_cc_cpu.model',
        ncores=4,
        device='cpu',
        dtype='float64',
        atomic_energies={},
    )

    structure_names = ['Ice_VIII']
    for structure_name in structure_names:
        pressures = [0.1] #MPa
        for pressure in pressures:
            coordinates = ['dh']
            for coordinate in coordinates:
                temperatures = [150] #K
                for temperature in temperatures:
                    path_opt = Path.cwd() / 'opt_structures_biased' / str('opt_' + structure_name + '_'+str(temperature)+'K_'+str(pressure)+'MPa.xyz')
                    info_str = coordinate + '_' + structure_name + '_'+str(temperature)+'K_'+str(pressure)+'MPa'
                    path_hessian = Path.cwd() / 'hessians_biased' / str('hessian_'+ info_str + '.npy')
                    
                    opt_geo = Geometry.load(path_opt)
                    hessian = np.load(path_hessian)

                    plumed_function = get_bias_function(pressure=pressure, temperature=temperature, number_of_atoms=len(opt_geo))
                    n_samples = 200
                    npt = True

                    ## generate nms samples in diff space and regular (rh) space, and energies of harmonic oscillator and mlp
                    tup5 = generate_nms_samples_energies(hessian,opt_geo,npt,n_samples,temperature,coo=coordinate)
                    samples_in_diff_coo, samples_in_r_h, e_harm, e_mlp, e_bias = tup5

                    np.save('energy_data/harm_energies_{}_T{}_coo{}_{}MPa'.format(structure_name,temperature,coordinate,pressure),np.array(e_harm))
                    np.save('energy_data/mlp_energies_{}_T{}_coo{}_{}MPa'.format(structure_name,temperature,coordinate,pressure),np.array(e_mlp))
                    np.save('energy_data/bias_energies_{}_T{}_coo{}_{}MPa'.format(structure_name,temperature,coordinate,pressure),np.array(e_bias))