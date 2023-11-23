import numpy as np
import mdtraj as md
import sys
import file_io as file_io
import periodic as periodic
import transformations as transformations
import matplotlib.pyplot as plt
import pickle
import geometry
plt.rcParams.update({'font.size':15})
# --------------------------------------------------------------------------------------------------------------------
# pickle utilities
# --------------------------------------------------------------------------------------------------------------------


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ---------------------------------------------------------------------------------------------------------------------
# index loading and processing
# ---------------------------------------------------------------------------------------------------------------------


def load_dummy_indices(file):
    ''' Assumes that bottom dummy section follows top dummy section immediately, and restarts numbering from 0'''
    indices = file_io.load_gromacs_index(file)
    inner_dummy_ind = np.array(indices['top_DUMY']) - indices['top_DUMY'][0]
    outer_dummy_ind = np.array(indices['bot_DUMY']) - indices['top_DUMY'][0]  # subtract from top! that's the 0 index
    return inner_dummy_ind, outer_dummy_ind


def load_leaflet_indices():
    pass


class mito_sectional_indices:
    def __init__(self, cylinder, junction,  flat):
        self.cylinder = cylinder
        self.junction = junction
        self.flat     = flat


def assign_dummy_particles_to_section(rho, z, mito_dims):
    in_cylinder, in_junction, in_flat = geometry.assign_to_mito_section(rho, z, mito_dims)
    return mito_sectional_indices(in_cylinder, in_junction, in_flat)


def load_lipid_indices(file, lipid_names):
    pass

# ---------------------------------------------------------------------------------------------------------------------
# load and process forces
# ---------------------------------------------------------------------------------------------------------------------


def load_raw_forces(path):
    return file_io.xvg_2_coords(file_io.load_large_text_file(path), 3)


# ---------------------------------------------------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------------------------------------------------

class mito_coordinates:
    ''' Compiled coordinate description of mito system. Can be used for dummy or lipid data'''
    def __init__(self, theta, rho, z, unified):
        self.theta = theta
        self.rho = rho
        self.z = z
        self.unified = unified


class raw_dummy_leaflet_data:
    def __init__(self, mito_coordinates, mito_shape, forces=None):
        # instance of mito_coordinates
        self.coordinates = mito_coordinates
        # instance of mito_shape
        self.mito_shape = mito_shape
        # n_frames * nparts * 3 array
        if forces is not None:
            self.forces = forces

dummy_leaflet_data = raw_dummy_leaflet_data

class processed_dummy_leaflet_data:
    def __init__(self, mito_coordinates, forces, force_errors):
        # instance of mito_coordinates
        self.coordinates = mito_coordinates
        # instance of mito_shape
        self.mito_shape = mito_shape
        # THIS IS TIME AVERAGED
        self.force = forces
        self.force_errors = force_errors


class lipid_leaflet_data:
    def __init__(self, mito_coordinates, lipid_indices, mito_shape):
        # instance of mito_coordinates
        self.coordinates = mito_coordinates
        # dict of indices
        self.lipid_indices = lipid_indices
        # instance of mito_shape
        self.mito_shape = mito_shape


# ----------------------------------------------------------------------------------------------------------------------
# general coordinate processing
# ---------------------------------------------------------------------------------------------------------------------


def cart_2_mito(coords, unitcell_lengths, mito_center):
    '''
        Does a cartesian to polar transformation on trajectory data, based on a given center point. Accounts for periodic
        boundaries by calling periodic.calc_vectors

    '''
    mito_center_scaled = mito_center[np.newaxis, :].repeat(coords.shape[1], axis=0)[np.newaxis, :, :]
    if coords.shape[0] > 1:
        mito_center_scaled = mito_center_scaled.repeat(coords.shape[0], axis=0)
    mito_vecs   = periodic.calc_vectors(mito_center_scaled, coords, unitcell_lengths)
    return transformations.cart2pol(mito_vecs.squeeze())


# ---------------------------------------------------------------------------------------------------------------------
# dummy loading functions
# ---------------------------------------------------------------------------------------------------------------------


def load_and_split_dummy_pdb(pdbpath, indexpath, mito_shape, zo):
    ''' Analyses should be split by leaflet, but it's easier with current setup to load the whole dummy system and then
        split into 2 leaflet data structures
    '''
    outer_mito_shape = geometry.mito_dims(mito_shape.l_cylinder,      mito_shape.r_cylinder + zo,
                                          mito_shape.r_junction - zo, mito_shape.l_flat)
    inner_mito_shape = geometry.mito_dims(mito_shape.l_cylinder,      mito_shape.r_cylinder - zo,
                                          mito_shape.r_junction + zo, mito_shape.l_flat)
    dummy_pdb = md.load(pdbpath)
    dummy_inner, dummy_outer = load_dummy_indices(indexpath)  # these are for dummy-only system, start at 0

    # center and transform whole system
    center = geometry.get_mito_center(dummy_pdb.xyz.squeeze(), mito_shape.l_cylinder)  # don't need inner/outer, this only uses l_cylinder
    theta, rho, z = cart_2_mito(dummy_pdb.xyz, dummy_pdb.unitcell_lengths, center)

    # unified coordinate is based on shape, so do separately
    inner_unified_coord = geometry.map_to_unified_coordinate(z[dummy_inner], rho[dummy_inner], inner_mito_shape)
    outer_unified_coord = geometry.map_to_unified_coordinate(z[dummy_outer], rho[dummy_outer], outer_mito_shape)

    inner_mito_coordinates =  mito_coordinates(theta.squeeze()[dummy_inner], rho.squeeze()[dummy_inner], z.squeeze()[dummy_inner], inner_unified_coord)
    outer_mito_coordinates =  mito_coordinates(theta.squeeze()[dummy_outer], rho.squeeze()[dummy_outer], z.squeeze()[dummy_outer], outer_unified_coord)
    return inner_mito_coordinates, outer_mito_coordinates


def load_and_split_dummy_forces(forcepath, indexpath):
    dummy_forces = load_raw_forces(forcepath)
    dummy_inner, dummy_outer = load_dummy_indices(indexpath)
    return dummy_forces[:, dummy_inner, :], dummy_forces[:, dummy_outer, :]


def load_all_dummy_info(pdbpath, forcepath, indexpath, mito_shape, zo):
    inner_coords, outer_coords = load_and_split_dummy_pdb(pdbpath, indexpath, mito_shape, zo)
    inner_forces, outer_forces = load_and_split_dummy_forces(forcepath, indexpath)

    outer_mito_shape = geometry.mito_dims(mito_shape.l_cylinder,      mito_shape.r_cylinder + zo,
                                          mito_shape.r_junction - zo, mito_shape.l_flat)
    inner_mito_shape = geometry.mito_dims(mito_shape.l_cylinder,      mito_shape.r_cylinder - zo,
                                          mito_shape.r_junction + zo, mito_shape.l_flat)

    return raw_dummy_leaflet_data(inner_coords, inner_mito_shape, forces=inner_forces), raw_dummy_leaflet_data(outer_coords, outer_mito_shape, forces=outer_forces)


# ---------------------------------------------------------------------------------------------------------------------
# force analysis
# ---------------------------------------------------------------------------------------------------------------------


def process_dummy_leaflet_forces(raw_dummy_data, bins, firstframe, lastframe):

    force_avg_per_bead = np.sqrt((raw_dummy_data.forces[firstframe:lastframe, :, :].mean(axis=0) ** 2).sum(axis=1))
    bin_assignments = np.digitize(raw_dummy_data.coordinates.unified, bins)
    forces_by_bin = np.zeros(len(bins) - 1)
    forces_errors_by_bin   = np.zeros(len(bins) - 1)
    for bin_ind in range(1, len(bins)):
        force_ind = bin_ind - 1
        forces_by_bin[force_ind] = force_avg_per_bead[bin_assignments == bin_ind].mean()
        forces_errors_by_bin[force_ind] = force_avg_per_bead[bin_assignments == bin_ind].std() / np.sqrt(force_avg_per_bead[bin_assignments == bin_ind].size)
    return forces_by_bin, forces_errors_by_bin


def process_dummy_system(raw_dummy_data, bin_spacing, firstframe=0, lastframe=None):
    if not lastframe:
        lastframe = raw_dummy_data.forces.shape[0]

    bins = np.arange(0, raw_dummy_data.coordinates.unified.max(), bin_spacing)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    force_means, force_errors = process_dummy_leaflet_forces(raw_dummy_data, bins, firstframe, lastframe=lastframe)
    z = geometry.unified_to_z(raw_dummy_data.mito_shape, bin_centers)
    rho = geometry.unified_to_rho(raw_dummy_data.mito_shape, bin_centers)
    bin_coords = mito_coordinates(0, rho, z, bin_centers)

    return processed_dummy_leaflet_data(bin_coords, force_means, force_errors)


# ---------------------------------------------------------------------------------------------------------------------
# coordinate analysis
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# main analysis
# -----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # mito geometry info
    dummy_zo = 5 / 2
    mito_shape = geometry.mito_dims(30, 10, 10, 56)
    inner_mito_shape = geometry.mito_dims(30, 10 - dummy_zo, 10 + dummy_zo, 56)
    outer_mito_shape = geometry.mito_dims(30, 10 + dummy_zo, 10 - dummy_zo, 56)

    # Shouldn't have to reload raw data every time - instead, load from pickle
    #load_raw = False
    load_raw = True
    if load_raw:
        
        PE_path = './'
        #print (PE_path)
        TEST_inner_dummy_data, TEST_outer_dummy_data = load_all_dummy_info('../dummy-ref.pdb', 'pt-force-4us.xvg',
                                                                           'index-bumpy.ndx', mito_shape, dummy_zo)
        print (TEST_inner_dummy_data)
        pickle_save(TEST_inner_dummy_data, "inner_dummy_data.pkl")
        pickle_save(TEST_outer_dummy_data, "outer_dummy_data.pkl")

    # analysis parameters
    firstframe = 750
    bin_width = 0.1

    TEST_inner_dummy_data = pickle_load('inner_dummy_data.pkl')
    TEST_outer_dummy_data = pickle_load('outer_dummy_data.pkl')
    TEST_inner_processed = process_dummy_system(TEST_inner_dummy_data, 1, firstframe=750)
    TEST_outer_processed = process_dummy_system(TEST_outer_dummy_data, 1, firstframe=750)


    def plot_processed_data(inner_data, outer_data, cmap='Reds'):
    #def plot_processed_data(inner_data, outer_data, cmax, cmap='Reds'):
        plt.figure()
        plt.scatter(inner_data.coordinates.rho, inner_data.coordinates.z, c=inner_data.force, cmap=cmap)
        plt.scatter(outer_data.coordinates.rho, outer_data.coordinates.z, c=outer_data.force, cmap=cmap)
        #plt.scatter(inner_data.coordinates.rho, inner_data.coordinates.z, vmin=0, vmax=cmax, c=inner_data.force, cmap=cmap)
        #plt.scatter(outer_data.coordinates.rho, outer_data.coordinates.z, vmin=0, vmax=cmax, c=outer_data.force, cmap=cmap)
        plt.xlabel("rho (nm)")
        plt.ylabel("z (nm)")
        plt.colorbar()
        plt.show()

    plot_processed_data(TEST_inner_processed, TEST_outer_processed)
    plt.figure()

