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

#Tropics are defined as equatorward of 30◦ latitude
#lat_bands = array([-90., -80., -70., -60., -50., -40., -30., -20., -10.,   0.,  10., 20.,  30.,  40.,  50.,  60.,  70.,  80.,  90.])
#zonal_weights = [1,1,1,1,1,1,2,2,2,
#                 2,2,2,1,1,1,1,1,1] #18 regions, 10 degrees latititude each
zonal_weights = [1,1,1,1,2,2,2,
                 2,2,2,1,1,1,1,1] #15 regions, 10 degrees latititude each

#regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
regional_weights = [1,1,1,
                   2,2,2] #6 regions


