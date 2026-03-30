import numpy as np  
from meteor import rsmap
from masking import support_from_masker
from meteor.sfcalc import gemmi_structure_to_calculated_map
import gemmi


def add_solvent(struc: gemmi.Structure, dmin :float, solvent_level: float=0.3) -> rsmap.Map:
    synth_map = gemmi_structure_to_calculated_map(struc, high_resolution_limit=dmin)
    synth_np = synth_map.to_3d_numpy_map(map_sampling=3)
    print(gemmi.AtomicRadiiSet.Cctbx)
    non_support_region = support_from_masker(
        pdb_file = struc,
        grid_shape= synth_np.shape,
        radii_set = gemmi.AtomicRadiiSet.Cctbx,
    )
    vacuum_region = np.logical_and(~non_support_region, solvent_level>synth_np)
    synth_np[vacuum_region] = solvent_level-synth_np[vacuum_region] 
    solvent_synth = rsmap.Map.from_3d_numpy_map(synth_np, high_resolution_limit=dmin, cell=synth_map.cell, spacegroup=synth_map.spacegroup)
    return solvent_synth, synth_map

