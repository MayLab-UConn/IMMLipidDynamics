import numpy as np


class mito_dims:
    ''' Helper structure for mito dimensions '''
    def __init__(self, l_cylinder, r_cylinder, r_junction, l_flat):
        self.l_cylinder = l_cylinder
        self.r_cylinder = r_cylinder
        self.r_junction = r_junction
        self.l_flat  = l_flat


def get_mito_center(coords, l_cylinder):
    '''
        For the z center, can just take the midpoint of the maximum and minimum coordinates, as the top and bottom
        plane z values are regular. For x/y, we have to potentially account for e.g. a cutoff of dummies along the x or
        y extreme that would throw off the COM by a tiny amount.

        Instead, we'll take the z com, and then center the x/y on the com of the cylindrical region. We don't need to
        worry about getting a little bit of the junction at the top and bottom of the cylinder - it's regular about the
        center just as the cylinder is.

        Parameters:
            coords - n_particles * 3 array
            dims   = mito_shape instance
        Returns the center as a np.array(3), doesn't make transformation
    '''
    z_com = (coords[:, 2].max() - coords[:, 2].min()) / 2.0
    xy_inrange = np.abs(coords[:, 2] - z_com) < l_cylinder / 2
    xy_means = coords[xy_inrange, 0:2].mean(axis=0)
    return np.array((xy_means[0], xy_means[1], z_com))


def assign_to_mito_section(rho, z, mito_shape):
    ''' For one frame, generates an assignment_indices instance assigning inner and outer indices to the three
        regions of the mitochondria

        This is based on a specific geometry of mitochondria, from mito dims

        Parameters
            rho - radial paramater (1D - n_parts)
            z   - z dimension      (1D - n_parts)
            mito_dims - mito shape object
        Returns
            in_cylinder - list of indices
            in_flat     - list of indices
            in_junction - list of indices. union of in_cylinder, in_flat, in_junction is 0 to rho.size - 1

    '''
    in_cylinder = np.abs(z) < (mito_shape.l_cylinder / 2)
    in_junction = (np.abs(z) >= (mito_shape.l_cylinder / 2)) & (rho <= (mito_shape.r_cylinder + mito_shape.r_junction))
    in_flat     = rho > (mito_shape.r_cylinder + mito_shape.r_junction)

    return in_cylinder, in_junction, in_flat


def map_to_cyl_regime(z, mito_shape):
    ''' The mapping of the cylindrical regime is just 0 = center, l_cylinder  / 2 = edge'''
    return np.abs(z)


def map_to_flat_regime(rho, mito_shape):
    ''' rho of 0 is the edge of the junction - no outer edge '''
    return rho - mito_shape.r_cylinder - mito_shape.r_junction


def map_to_junction_regime(z, rho, mito_shape):
    ''' Angular description only to start - as a distance based one would require averaging the radii of all particles
       in the description. 0 degrees  = cylinder end, 90 degrees = flat end
    '''

    # first center coordinates relative to "circle" that describes the junction
    centered_z = np.abs(z) - (mito_shape.l_cylinder / 2)
    centered_rho = rho - mito_shape.r_cylinder - mito_shape.r_junction

    # coordinates are now in second quadrant of circle, subtract by 90 to get to convention
    # remember than arctan2 goes (y, x)
    return 180 - np.arctan2(centered_z, centered_rho) * 180 / np.pi


def calc_inner_toroidal_area_by_angle(r_torus, r_tube, theta):
    return 2 * np.pi * r_tube * r_torus * theta - 2 * np.pi * r_tube * r_tube * np.sin(theta)


def map_to_unified_coordinate(z, rho, mito_shape):
    in_cyl, in_junc, in_flat = assign_to_mito_section(rho, z, mito_shape)

    junc_offset = mito_shape.l_cylinder / 2
    flat_offset = junc_offset + np.pi * mito_shape.r_junction / 2

    unified_coordinate = np.zeros(z.shape)
    unified_coordinate[in_cyl] = np.abs(z[in_cyl])
    unified_coordinate[in_junc] = map_to_junction_regime(z[in_junc], rho[in_junc], mito_shape) * mito_shape.r_junction / (180 / np.pi) + junc_offset  # noqa
    unified_coordinate[in_flat] = map_to_flat_regime(rho[in_flat], mito_shape) + flat_offset

    return unified_coordinate


def circle_segment_area(r, t):
    ''' r is circle radius, t is distance from center to perpindicular chord'''
    return  (r ** 2) * np.arccos(t / r) - t * np.sqrt(r ** 2 - t ** 2)


def unified_to_junction_angle(mito_dims, unified_coordinate):
    arc_length = unified_coordinate - mito_dims.l_cylinder / 2
    return arc_length / mito_dims.r_junction


def unified_to_flat_radius(mito_dims, unified_coordinate):
    return mito_dims.r_junction + mito_dims.r_cylinder + (unified_coordinate - mito_dims.l_cylinder / 2 - mito_dims.r_junction * np.pi / 2)


def unified_to_rho(mito_dims, unified_coordinate):
    rho = np.zeros(unified_coordinate.size) + mito_dims.r_cylinder

    injuncregion = (unified_coordinate > mito_dims.l_cylinder / 2) & (unified_coordinate <= mito_dims.l_cylinder / 2 + mito_dims.r_junction * np.pi / 2)
    inflatregion = unified_coordinate > (mito_dims.l_cylinder / 2 + mito_dims.r_junction * np.pi / 2)

    j_angle = unified_to_junction_angle(mito_dims, unified_coordinate[injuncregion])
    rho[injuncregion] = rho[injuncregion] + mito_dims.r_junction * (1 - np.cos(j_angle))

    rho[inflatregion] = unified_to_flat_radius(mito_dims, unified_coordinate[inflatregion])
    return rho


def unified_to_z(mito_dims, unified_coordinate):
    z = np.zeros(unified_coordinate.size)
    incylregion = unified_coordinate <= (mito_dims.l_cylinder / 2)
    inflatregion = unified_coordinate >= mito_dims.l_cylinder / 2 + mito_dims.r_junction * np.pi / 2
    injuncregion = ((~incylregion) & (~ inflatregion))

    z[incylregion] = unified_coordinate[incylregion]
    j_angle = unified_to_junction_angle(mito_dims, unified_coordinate[injuncregion])
    z[injuncregion] = mito_dims.l_cylinder / 2 + mito_dims.r_junction * np.sin(j_angle)

    z[inflatregion] = mito_dims.l_cylinder / 2 + mito_dims.r_junction
    return z


def calc_cylindrical_section_area(mito_dims, bin_edge_1, bin_edge_2):
    ''' Just takes simple difference in l dimension between edges '''
    return 2 * np.pi * mito_dims.r_cylinder * (bin_edge_2 - bin_edge_1)


def calc_junction_section_area(mito_dims, bin_angle_1, bin_angle_2):
    ''' Calculate the two partial areas and subtract. Bin angles are 0 for the cylindrical end, pi/2 for the flat end'''
    r_torus = mito_dims.r_cylinder + mito_dims.r_junction
    r_tube = mito_dims.r_junction
    return calc_inner_toroidal_area_by_angle(r_torus, r_tube, bin_angle_2) - calc_inner_toroidal_area_by_angle(r_torus, r_tube, bin_angle_1)


def calc_flat_section_area(mito_dims, bin_r_1, bin_r_2):
    ''' Bin inputs should be radii'''
    if bin_r_1 > mito_dims.l_flat / 2:
        coord1_excluded = 4 * circle_segment_area(bin_r_1, mito_dims.l_flat / 2)
    else:
        coord1_excluded = 0

    if bin_r_2 > mito_dims.l_flat / 2:
        coord2_excluded = 4 * circle_segment_area(bin_r_2, mito_dims.l_flat / 2)
    else:
        coord2_excluded = 0

    return (np.pi * (bin_r_2 ** 2) - coord2_excluded) - (np.pi * (bin_r_1 ** 2) - coord1_excluded)


def get_unified_coord_section(mito_dims, unified_coordinate):
    if unified_coordinate <= mito_dims.l_cylinder / 2:
        return 'cyl'
    elif unified_coordinate >= mito_dims.l_cylinder / 2 + np.pi * mito_dims.r_junction  / 2:
        return 'flat'
    else:
        return 'junc'


def calc_unified_section_area(mito_dims, coord1, coord2):
    coord1_section = get_unified_coord_section(mito_dims, coord1)
    coord2_section = get_unified_coord_section(mito_dims, coord2)

    if coord1_section == 'cyl':
        if coord2_section == 'cyl':
            return calc_cylindrical_section_area(mito_dims, coord1, coord2)
        elif coord2_section == 'junc':
            # get cylindrical area up to junction border, then junction area from 0 to coord 2
            cyl_region = calc_cylindrical_section_area(mito_dims, coord1, mito_dims.l_cylinder / 2)
            coord2_angle = unified_to_junction_angle(mito_dims, coord2)
            return cyl_region + calc_junction_section_area(mito_dims, 0, coord2_angle)
        else:
            raise NotImplementedError("cyl to flat area not implemented")
    elif coord1_section == 'junc':
        coord1_angle = unified_to_junction_angle(mito_dims, coord1)
        if coord2_section == 'junc':
            coord2_angle = unified_to_junction_angle(mito_dims, coord2)
            return calc_junction_section_area(mito_dims, coord1_angle, coord2_angle)
        elif coord2_section == 'flat':
            # calculate junction area to flat border
            junc_area = calc_junction_section_area(mito_dims, coord1_angle, np.pi / 2)
            # add flat area from border, which is r_junc + r_cyl
            coord2_radius = unified_to_flat_radius(mito_dims, coord2)
            return junc_area + calc_flat_section_area(mito_dims, mito_dims.r_junction + mito_dims.r_cylinder, coord2_radius)
    elif coord1_section == 'flat':
        return calc_flat_section_area(mito_dims, unified_to_flat_radius(mito_dims, coord1), unified_to_flat_radius(mito_dims, coord2))
    else:
        raise Exception("coordinate section not defined")
