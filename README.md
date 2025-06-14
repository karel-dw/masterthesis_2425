Scripts for masterthesis Karel De Witte at CMM 2024-2025

Three files contain the main workflow of the novel thermodynamic integration scheme (see figure 4.1 of the thesis).
These three files are:
  optimization.py
  extended_hessian.py
  md_mixed.py
  
Three files support this workflow, namely 
  transform.py contains the transformations to and from scaled/deformed coordinates,
  G_ref_formula.py contains the analytical expression for the Gibbs free energy of the reference NPT harmonic approximation,
  and extended_harmonic_hamiltonian.py contains the code to extract energy/forces/stresses from the NPT harmonic approximation (this script only works integrated in psiflow).
  
One file contains the script of NPT normal mode sampling, namely
  NPT_NMS.py
