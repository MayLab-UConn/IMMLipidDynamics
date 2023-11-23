import numpy as np

'''
    Contains scripts relating to calculating simulation observables while accounting for periodic boundaries. The most
    import one is calculating vectors, as most other things (distances, angles) etc can be derived from them.

'''


def calc_vectors(p_origin, p_destination, boxdims):
    """
        MDtraj has functionality for computing distances but it's not always applicable to every dataset, and distances
        contain no directonality. This function will calculate vectors for coordinates, taking into account the box
        dimensions. For simplicity, will only take in mdtraj xyz shaped arrays (and trajectory.unitcell_lengths)

        Note that this will only calculate vectors within 1 periodic image!

        Parameters
            p_origin      - n_frames * n_particles * n_dimensions coordinate array
            p_destination - n_frames * n_particles * n_dimensions coordinate array - same size as p_origin
            boxdims       - n_frames * n_dimensions array of box dimensions

        Returns
            vecs -n_frames * n_particles * n_dimensions array
    """
    if not p_origin.ndim == 3:
        raise ValueError("coordinates should be nframes * nparticles * ndims, p_origin shape = {}".format(p_origin.shape))   # noqa
    if not boxdims.ndim == 2:
        raise ValueError("boxdims should be nframes * nparticles, boxdims shape = {}".format(boxdims.shape))
    if not p_origin.shape == p_destination.shape:
        raise ValueError("input vector dimension mismatch. Origin shape = {}, destination shape =  {}".format(
                         p_origin.shape, p_destination.shape))
    if not p_origin.shape[0] == boxdims.shape[0]:  # mismatch between number of frames in coords and boxdims
        raise ValueError("Mismatch between number of frames in coordinates ({}) and boxdims ({})".format(
                         p_origin.shape[0], boxdims.shape[0]))
    if not p_origin.shape[2] == boxdims.shape[1]:  # mismatch between dimensionality
        raise ValueError("Mismatch between number of dimensions in coordinates ({}) and boxdims ({})".format(
                         p_origin.shape[2], boxdims.shape[1]))

    boxdims_reshaped = boxdims[:, np.newaxis, :]  # allows broadcasting
    boxdims_midpoint = boxdims_reshaped / 2
    vecs = p_destination - p_origin
    veclengths = np.abs(vecs)

    # these are the vectors who's periodic image are closer than the original vecotor
    vecs_gt_boxdims = veclengths >  (boxdims_midpoint)  # these positions will be changed

    # boolean arrays for identifying closest periodic image - based on vector direction instead of
    # place in box, which might not be centered on (0, 0, 0)
    negative_vecs = vecs < 0
    positive_vecs = vecs > 0

    # for positive vectors greater than half the box, use previous periodic image
    vecs[vecs_gt_boxdims & positive_vecs] = -(boxdims_reshaped - veclengths)[vecs_gt_boxdims & positive_vecs]

    # for negative vectors greater than half the box, use next periodic image.
    vecs[vecs_gt_boxdims & negative_vecs] = (boxdims_reshaped - veclengths)[vecs_gt_boxdims & negative_vecs]

    return vecs
