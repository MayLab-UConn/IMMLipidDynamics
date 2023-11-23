import numpy as np


def xvg_2_coords(xvg_input, dims, return_time_data=False):
    '''
        Processes a numpy array of xvg data into mdtraj style formatting. Strips first column (time usually)
        optionally returns it
    '''
    reshaped_data = xvg_input[:, 1:].reshape(xvg_input.shape[0], int((xvg_input.shape[1] - 1) / dims), dims)
    if return_time_data:
        times = xvg_input[:, 0]
        return reshaped_data, times
    else:
        return reshaped_data


def load_xvg(file, comments=('#', '@'), dims=3, return_time_data=False):
    ''' Loads an xvg file, created from gromacs. For a typical gromacs-derived xvg giving information on
        n particles with m dimensions to the data, the format is c1=time, c2 to c2 + m = data on first particle, and
        so on. Each row is a time point. Will throw out time point unless specified

        Parameters
            file             - path to xvg file
            comments         - comments to ignore from header. "#" and "@" typically handle it
            dims             - dimensions of data. Ie for every particle, how many pieces of information
            return_time_data - boolean. If true, returns a tuple of (data, time)
        Returns
            data             - nframes * nparticles * ndims
            times            - (optional) nframes array of times, same units as in xvg file
    '''

    data = np.loadtxt(file, dtype=float, comments=comments)
    if (data.shape[1] - 1) % dims > 0:
        raise ValueError("(dims * n_particles) + 1 does not equal number of columns in xvg")

    return xvg_2_coords(data, dims, return_time_data=return_time_data)


def load_large_text_file(file, delimiter=' ', verbose=True, dtype=float, comments=('@', '#')):
    '''
        Numpy.loadtxt has some memory problems on large files. This function will likely be slow but is more memory
        efficient. Loops through file twice - once to get dimensions, another to load array
    '''
    n_columns = 0
    n_rows    = 0
    with open(file, 'r') as fin:
        if verbose:
            print("opening file for initial read")
        for line in fin:
            # skip comment line
            if not line[0] in comments and line.strip():
                # if first data row, get dimensions
                if not n_rows:
                    n_columns = np.fromstring(line, sep=delimiter).size
                n_rows += 1

    if not n_columns or not n_rows:
        raise Exception("file error - file contained {} rows and {} columns".format(n_rows, n_columns))
    else:
        output = np.zeros((n_rows, n_columns), dtype=dtype)
        row = 0
        with open(file, 'r') as fin:
            if verbose:
                print("opening file for assignment read - nrows = {}, ncolumns = {}".format(n_rows, n_columns))
            for line in fin:
                if not line[0] in comments and line.strip():
                    line_array = np.fromstring(line, sep=delimiter, dtype=dtype)
                    if line_array.size == n_columns:
                        output[row] = line_array
                        row += 1
                    else:
                        print(line[:80])
                        raise Exception("Data inconsistency at row {}, expected {} columns, got {}".format(row,
                                        n_columns, line_array.size ))
    return output


def load_gromacs_index(index_file):
    ''' Loads a gromacs style index file. Decrements all read indices by 1, as numbering starts at 1 in the files, but
        we'll be using these as array indices

        Parameters -
            index_file - path to a file
        Returns -
            index_dict - dictionary of index string : list of integer values
    '''
    with open(index_file, 'r') as fin:
        index_dict = {}
        curr_group = []
        curr_nums = []
        for line in fin:

            # check for opening and closing brackets
            if "[" in line and "]" in line:

                # add previous to dictionary only if one existed before - accounts for initial case
                if curr_group:
                    index_dict[curr_group] = curr_nums

                # reset group and index count
                curr_group = line.split("[", 1)[-1].split("]", 1)[0].strip()
                curr_nums = []
            elif curr_group:
                curr_nums += [int(i) - 1 for i in line.split()]    # decrement each one
        # one last time
        if curr_nums:
            index_dict[curr_group] = curr_nums
    return index_dict
