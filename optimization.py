import psiflow
from psiflow.hamiltonians import MACEHamiltonian, PlumedHamiltonian
from pathlib import Path
from parsl import File
from psiflow.geometry import Geometry
from ase.units import bar, kB
from psiflow.sampling.ase import optimize
from psiflow.data.utils import write_frames
from parsl.data_provider.files import File


'''
Use Pidobbels ase-optim branch to perform ASE optimizations !!!
'''

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



def main(structure_name, pressure, temperature):
    path_structure = Path.cwd() / 'input_structures' / str(structure_name + '.xyz')
    #path_structure = str('opt_structures_biased/opt_' + structure_name + '_200K_0.1MPa.xyz')
    geometry = Geometry.load(path_structure)

    mace = MACEHamiltonian(File(Path.cwd() / "ani500k_cc_cpu.model"), {})
    bias = get_bias(pressure=pressure, temperature=temperature, number_of_atoms=len(geometry))
    hamiltonian = mace + bias

    filename = str('opt_'+structure_name+'_'+str(temperature)+'K_'+str(pressure)+'MPa.xyz')
    path_opt_structure = Path.cwd() / 'opt_structures_biased' / filename
    if not Path(path_opt_structure).exists():
        opt_geo = optimize(geometry, hamiltonian, f_max=1e-3, mode='full', pressure = 0)
        write_frames(opt_geo, outputs = [File(path_opt_structure)])

if __name__ == '__main__':
    with psiflow.load():
        for phase in ['Ice_VIII','Ice_XI']:
            for P in [0.1]:         # in MPa
                for T in [150]:     # in K
                    main(phase,P,T)
    print('Done')