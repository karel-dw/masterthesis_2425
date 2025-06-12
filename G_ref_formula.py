from psiflow.geometry import Geometry
import numpy as np
from ase.units import kB
from psiflow.utils.apps import get_attribute

def extended_harmonic_free_energy(
        opt_geo: Geometry,
        hessian: np.ndarray,
        temperature: float,
) -> float:
    assert np.allclose(hessian,hessian.T), 'hessian should be symmetric'
    eigvals = np.linalg.eigvalsh(hessian)
    assert np.min(eigvals) > -0.01, 'negative eigen value'

    h = 4.1356677e-15  #_hplanck*J  --> h in eV*s
    T = temperature
    beta = 1/(kB*T)
    V_0 = np.linalg.det(opt_geo.cell)
    N = len(opt_geo.per_atom.positions)
    D = eigvals[6:]
    energy_0 = get_attribute(opt_geo, "energy")

    masses = opt_geo.atomic_masses
    sum_ln_masses = 0
    for m in masses:
        sum_ln_masses += np.log(m)
    
    sum_ln_D = 0
    for Di in D:
        sum_ln_D += np.log(Di)

    t1 = energy_0
    t2 = N/beta*np.log(V_0)   # to power N, and not to power 2, because energy_0 contains -(N-2)/beta*ln(V_0)
    t3 = -3/(2*beta)*sum_ln_masses
    t4 = -(3*N/beta)*np.log(np.sqrt(2*np.pi*kB*T)/h)
    t5 = (1/(2*beta))*sum_ln_D
    t6 = -(3*N+3)/(2*beta)*np.log(2*np.pi*kB*T)

    G_ref = t1+t2+t3+t4+t5+t6
    print('Gref')
    return G_ref