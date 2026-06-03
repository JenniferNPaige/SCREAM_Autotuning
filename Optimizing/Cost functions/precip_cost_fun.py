#variable weighting
var_weights_dict = {
    'PCP': 0.4,
    'TLWP': 0.2,
    'OSR': 0.2,
    'OLR': 0.2
    }

#zonal, regional, global weighting
zrg_weights_dict = {
    'zonal': (1/3),
    'regional': (1/3),
    'global': (1/3)
    } 

#summer/winter weights
DY_weights_dict = {
    'DY1': (1/2),
    'DY2': (1/2)
    } 

#weighted averaging
zonal_weights = None
regional_weights = None