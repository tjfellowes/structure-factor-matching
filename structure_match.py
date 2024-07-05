import io
from pathlib import Path
from multiprocessing import Process
import re

import cctbx
from cctbx.xray.structure_factors.manager import managed_calculation_base
from libtbx import adopt_init_args
import scipy.spatial.distance
import iotbx.cif
import iotbx.reflection_file_reader

class StructureMatch():
    '''
    '''
    # Monkey patch the structure factor calculation method to remove the check for metric symmetry.
    managed_calculation_base.__init__ = lambda self, manager, xray_structure, miller_set, algorithm: adopt_init_args(self, locals(), hide=True)

    def __init__(self):
        pass

    def multi_addsym(cif_files):
        processes = [(i.name, Process(target = run_addsym, args=(i,))) for i in cif_files if not Path(str(i.parent) + '\\' + str(i.stem + '_pl.res')).exists()]
        for name, process in processes:
            print(name, process)
            process.start()
        for name, process in processes:
            process.join()

    def cosine_distance(sf1, sf2):
        '''
        Calculates the cosine distance between two sets of structure factors.

            Parameters:
                sf1 (cctbx.miller.array): The first set of structure factors
                sf2 (cctbx.miller.array): The second set of structure factors
            
            Returns:
                distance (float): cosine distance between two sets of structure factors
        '''
        return scipy.spatial.distance.cosine(*[i.data() for i in sf1.common_sets(sf2)])

    def r_factor_distance(sf1, sf2):
        return sf2.r1_factor(sf1)

    def braycurtis_distance(sf1,sf2):
        return scipy.spatial.distance.braycurtis(*[i.data() for i in sf1.common_sets(sf2)])

    def canberra_distance(sf1,sf2):
        return scipy.spatial.distance.canberra(*[i.data() for i in sf1.common_sets(sf2)])

    def chebyshev_distance(sf1,sf2):
        return scipy.spatial.distance.chebyshev(*[i.data() for i in sf1.common_sets(sf2)])

    def cityblock_distance(sf1,sf2):
        return scipy.spatial.distance.cityblock(*[i.data() for i in sf1.common_sets(sf2)])

    def euclidean_distance(sf1,sf2):
        return scipy.spatial.distance.euclidean(*[i.data() for i in sf1.common_sets(sf2)])

    def jensenshannon_distance(sf1,sf2):
        return scipy.spatial.distance.jensenshannon(*[i.data() for i in sf1.common_sets(sf2)])

    def minkowski_distance(sf1,sf2):
        return scipy.spatial.distance.minkowski(*[i.data() for i in sf1.common_sets(sf2)])

    def compare_cifs(reference_cif_file, comparison_cif_file, similarity_function=cosine_distance, resolution_limit=1):
        """
        Calculates the similarity between a set of reference structure factors and those derives from one or more comparison structures.

            Parameters:
                    reference_cif_file (pathlib.Path): Reference cif
                    comparison_cif_file (list(pathlib.Path)): Structural model to compare against.
                    similarity_function (function(sf1, sf2)): function to calculate the similarity, cosine distance by default.
                    resolution_limit (float): only compare reflections up to this d spacing.
            Returns:
                    distance (float): Distance between the reference structure and the comparison structure.
        """
        reference = dict()
        reference["name"], reference["structure"] = (
            iotbx.cif.reader(str(reference_cif_file))
            .build_crystal_structures()
            .popitem()
        )

        reference["reflections"] = (
            reference["structure"]
            .structure_factors(anomalous_flag=False, d_min=resolution_limit)
            .f_calc()
            .as_amplitude_array()
        )

        comparison = dict()
        comparison["name"], comparison["structure"] = (
            iotbx.cif.reader(str(comparison_cif_file))
            .build_crystal_structures()
            .popitem()
        )

        comparison["reflections"] = (
            comparison["structure"]
            .structure_factors(anomalous_flag=False, d_min=resolution_limit)
            .f_calc()
            .as_amplitude_array()
        )

        return similarity_function(reference['reflections'], comparison['reflections'])

    def generate_hkl_file(cif_file, hkl_file='shelx.hkl', resolution_limit=1):
        """
        Generates a shelx style hkl file for a given structural model

            Parameters:
                    cif_file (pathlib.Path): cif file containing structural model
                    resolution_limit (float): only generate reflections up to this d spacing.
            Returns:
                    filename (string): filename of the generated hkl file
        """
        reference = dict()
        reference["name"], reference["structure"] = (
            iotbx.cif.reader(str(cif_file)).build_crystal_structures().popitem()
        )

        reference["reflections"] = (
            reference["structure"]
            .structure_factors(anomalous_flag=False, d_min=resolution_limit)
            .f_calc()
            .as_amplitude_array()
        )

        with open(hkl_file, 'w') as f:
            reference["reflections"].export_as_shelx_hklf(file_object=f)

        return(hkl_file)

    def compare_cif_to_hkl(reference_hkl_file, reference_cif_file, comparison_cif_file, similarity_function=cosine_distance, resolution_limit=(999,1)):
        '''
        Calculates the similarity between a set of reference structure factors and those derives from one or more comparison structures.

            Parameters:
                    reference_hkl_file (pathlib.Path): Reference structure factor file
                    reference_cif_file (pathlib.Path): Reference cif containing symmetry information for reference_hkl (if it contains a structural model it will be ignored)
                    comparison_cif_file (pathlib.Path): Structural models to compare against.
                    similarity_function (function(sf1, sf2)): function to calculate the similarity, cosine distance by default.
                    resolution_limit ((float, float)): only compare reflections between these limits.
            Returns:
                    
        '''

        reference = {}

        # Add a lonely oxygen atom to the reference cif file so cctbx will read it properly. A dirty hack.
        with open(reference_cif_file, 'r') as f:
            with io.StringIO() as tmp:
                f.seek(0)
                tmp.write(f.read())
                f.seek(0)
                if '_atom_site_label' not in f.read():
                    tmp.write('''
loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_U_iso_or_equiv
  O O 0.0 0.0 0.0 1
''')
                tmp.seek(0)

                # Read symmetry of reference structure from cif file
                reference['name'], reference['structure'] = iotbx.cif.reader(file_object=tmp).build_crystal_structures().popitem()

        # Read reference structure factors from hkl file, merge, res limit, and map to niggli cell
        reference['reflections'] = (iotbx.reflection_file_reader
        .any_reflection_file(str(reference_hkl_file) + '=hklf4')
        .as_miller_arrays(reference['structure'])[0].as_amplitude_array().resolution_filter(*resolution_limit))

        comparison = {}

        comparison["name"], comparison["structure"] = (
            iotbx.cif.reader(str(comparison_cif_file))
            .build_crystal_structures()
            .popitem()
        )

        is_similar = reference['reflections'].is_similar_symmetry(comparison['structure'], absolute_angle_tolerance=90, absolute_length_tolerance=100)
        is_similar = True
        if is_similar:
            comparison['reflections'] = reference['reflections'].structure_factors_from_scatterers(comparison['structure']).f_calc().as_amplitude_array()
            distance = similarity_function(comparison['reflections'], reference['reflections'])

        return(distance)
