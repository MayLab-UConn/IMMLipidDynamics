# IMMLipidDynamics
Supporting Data is provided at https://github.com/MayLab-UConn/IMMLipidDynamics for manuscript "Curvature Sensing Lipid Dynamics in a Mitochondrial Inner Membrane Model" by V.K. Golla, K.J. Boyd and E.R. May, to be published in Communications Biology 2023/2024.

Data Organization:
GMX_Input_files:  Contains simulation input files (.mdp, .top, .gro) for POPC/Test system. Equilibration (step6.*.mdp) and production (md.mdp) are provided. toppar subdirectory contains .itp files for systems as well as CDL-1 and CDL-2 .itp files with 4 beads/acyl chain.

IMM_analysis: Contains example data for the POPC/Test system. Only the PO4 bead positions are present in the .xtc and .gro files and the leaflets are separated. These trajectories can be analyzed for force calucations, lipid accumulation and compartmental analysis using the subdirectory python scripts.

Figure_data: Contains data for main text figures 2,4-10. 
