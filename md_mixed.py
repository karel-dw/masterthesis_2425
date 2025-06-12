import psiflow
from parsl.dataflow.futures import AppFuture
from parsl import python_app, File
from psiflow.hamiltonians import MACEHamiltonian, MixtureHamiltonian, Harmonic, ExtendedHarmonic, PlumedHamiltonian
from psiflow.geometry import Geometry
import numpy as np
from psiflow.sampling import Walker, sample, SimulationOutput
import pickle
from pathlib import Path
from ase.units import bar, kB


PLUMED_INPUT = """UNITS LENGTH=A ENERGY=eV
vol: VOLUME
"""

def get_bias(pressure: float, temperature: float, number_of_atoms: int) -> PlumedHamiltonian:
    c1 = pressure * 10 * bar  # Convert to ase units
    c2 = kB * temperature * (number_of_atoms - 2)

    plumed_str = PLUMED_INPUT
    plumed_str += '\n'
    plumed_str += f'pv: MATHEVAL ARG=vol VAR=x FUNC={c1}*x PERIODIC=NO  \n'
    plumed_str += f'lnv: MATHEVAL ARG=vol VAR=x FUNC={c2}*log(x) PERIODIC=NO  \n'   # not easy to find if log(x) calculates ln(x)
    plumed_str += f'bv: MATHEVAL ARG=pv,lnv VAR=x,y FUNC=x-y PERIODIC=NO  \n'
    plumed_str += 'BIASVALUE ARG=bv\n'
    return PlumedHamiltonian(plumed_str)


@python_app(executors=["default_threads"])
def save_simulation_output(output: SimulationOutput, outputs: list[File]) -> AppFuture:
    """"""
    # TODO: this function internally works with futures.. why does this work?
    from psiflow.sampling.output import potential_component_names

    # evaluate futures
    data = {k: v.result() for k, v in output._data.items()}

    # rename hamiltonian contributions
    names_ipi = potential_component_names(len(output.hamiltonians))
    names = [h.__class__.__name__ for h in output.hamiltonians]
    for name_ipi, name in zip(names_ipi, names):
        data[name] = data.pop(name_ipi)


    pickle.dump(data, open(outputs[0], 'wb'))

    if len(outputs) == 2:
        output.trajectory.save(outputs[1].filepath)

    return


def main():
    structure_name = 'Ice_VIII'
    coord_system = "dh"
    temperature = 100  # in K
    pressure = 0.1    # in MPa
    path_opt_bias = Path.cwd() / 'opt_structures_biased' / str('opt_' + structure_name + '_'+str(temperature)+'K_'+str(pressure)+'MPa.xyz')
    path_hessian_bias = Path.cwd() / 'hessians_biased' / str('hessian_'+ coord_system + '_' + structure_name + '_'+str(temperature)+'K_'+str(pressure)+'MPa.npy')

    opt_geo_bias = Geometry.load(path_opt_bias)
    extended_hessian_bias = np.load(path_hessian_bias)

    eigvals = np.linalg.eigvals(extended_hessian_bias)
    assert np.min(eigvals) > -0.01, 'negative eigen value'
    if np.max(np.iscomplex(eigvals)):
        print('Complex eigenvalues!!!')
        print('max asymmetry: {}'.format(np.max(np.abs(extended_hessian_bias - extended_hessian_bias.T))))
    extended_hessian_bias = np.copy((extended_hessian_bias + extended_hessian_bias.T)/2)

    extended_harmonic_bias = ExtendedHarmonic(
        reference_geometry=opt_geo_bias,
        extended_hessian=extended_hessian_bias,
        hessian_coordinates = coord_system,
    )
    mace = MACEHamiltonian(File(Path.cwd() / "ani500k_cc_cpu.model"), {})

    bias = get_bias(pressure=pressure, temperature=temperature, number_of_atoms=len(opt_geo_bias))

    for lmd in [0.0]: # , 0.5
        mixture : MixtureHamiltonian = (1-lmd) * (extended_harmonic_bias - bias) + lmd * mace
        walker = Walker(opt_geo_bias, hamiltonian=mixture, temperature=temperature, pressure=pressure)
        outputs = sample(
            walkers=[walker],
            steps=1500,
            step=5,
            start=0,
            keep_trajectory=True,
            observables=['cell_h{angstrom}','pressure_md{megapascal}', 'stress_md{megapascal}'],
            fix_com=True,
            use_unique_seeds=True,
        )

        pickle_path = "pickle_data/test_output_"+coord_system + '_' + structure_name + '_'+str(temperature)+'K_'+str(pressure)+'MPa'+"_"+str(lmd)+".pickle"
        trajectory_path = "trajectory/test_trajectory_"+coord_system + '_' + structure_name + '_'+str(temperature)+'K_'+str(pressure)+'MPa'+"_"+str(lmd)+".xyz"
        save_simulation_output(outputs[0], outputs=[File(pickle_path), File(trajectory_path)])

if __name__ == '__main__':
    with psiflow.load():
        main()
    print('Done')