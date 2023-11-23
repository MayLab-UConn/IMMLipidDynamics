import numpy as np


def cart2pol(cart_coords):
    ''' Converts cartesian to polar coordinates. Acts on first 2 columns (xy)
        returns tuple of (theta,rho,z)

        Parameters -
            cart_coords - n_parts * 3 array
        returns
            theta - n_parts, angular coordinates  - range is -pi to pi
            rho   - n_parts, radial coordinates
            z     - n_parts, z coordinates of original array
    '''

    # input validation
    # if len(cart_coords.shape) is not 2 or cart_coords.shape[1] is not 3:
    #    raise ValueError("dimension mismatch, expected (n_particles * 3), got {}".format(cart_coords.shape))
    if len(cart_coords.shape) == 2:
        x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
    elif len(cart_coords.shape) == 3:
        x, y, z = cart_coords[:, :, 0], cart_coords[:, :, 1], cart_coords[:, :, 2]

    theta    = np.arctan2(y, x)
    rho  = np.sqrt(x**2 + y**2)
    return (theta, rho, z)


def cart2spherical(cart_coords):
    ''' Converts cartesian to spherical coordinates.

        Parameters -
            cart_coords - n_parts * 3 array
        returns
            theta - n_parts, - range is -pi to pi
            rho   - n_parts, radial coordinates
            phi   - n_parts  - range is 0 to pi
    '''

    # input validation
    if len(cart_coords.shape) is not 2 or cart_coords.shape[1] is not 3:
        raise ValueError("dimension mismatch, expected (n_particles * 3), got {}".format(cart_coords.shape))

    x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
    theta    = np.arctan2(y, x)
    rho  = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arccos(z / rho)
    return (theta, rho, phi)
