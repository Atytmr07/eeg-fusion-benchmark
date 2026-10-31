# Sensitivity of artefact removal to rounding-level perturbations

Each recording cleaned, then cleaned again 3 times after multiplying every sample by (1 + 1e-07 N(0, 1)). 'changed': relative RMS difference above 1e-3 or different components removed.

                perturbations  recordings  changed  max_rms_diff_rel  min_corr  max_abs_dkept
method                                                                                       
gedai                      27           9       16          1.742684 -0.122780       0.757574
gedai-fullrank             27           9       11          1.250502  0.350921       0.614386
infomax                    27           9        7          0.826600  0.265678       0.443451
