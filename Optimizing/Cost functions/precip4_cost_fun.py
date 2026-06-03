#variable weighting
var_weights_dict = {
    'PCP': (4/7),
    'TLWP': (1/7),
    'OSR': (1/7),
    'OLR': (1/7)
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

#weighted averaging - defaults to even
zonal_weights = None
regional_weights = None