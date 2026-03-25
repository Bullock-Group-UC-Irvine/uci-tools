def save_rotation_matrix_data(test_data_dir):
    '''
    Generate expected rotation matrix values using the old
    vrrotvec/vrrotvec2mat implementation and save them to an HDF5
    file for use in tests that verify backward compatibility after
    the rotation_matrix_fr_vecs rewrite.

    This function omits the parallel and antiparallel cases. For
    both, vrrotvec returns a zero-length rotation axis, which causes
    vrrotvec2mat to divide by zero and produce NaN. Unit tests for
    those degenerate cases verify only that the new implementation
    satisfies the required mathematical properties (R @ a_hat ==
    b_hat, R.T @ R == I, det(R) == 1).

    Run this utility once before changing rotate_galaxy.py, then
    commit the generated HDF5 file alongside the new tests.

    Usage
    -----
    from uci_tools.tests.utils import save_rotation_matrix_data
    save_rotation_matrix_data('uci_tools/tests/test_data')
    '''
    import os
    import h5py
    import numpy as np
    from uci_tools.rotate_galaxy import vrrotvec, vrrotvec2mat

    # Each tuple: (dataset_name, source_vector_a, target_vector_b).
    # The old code normalizes internally through vrrotvec, so we
    # pass the vectors as-is and normalize before storing so the
    # tests work with unit vectors.
    test_cases = [
        (
            'x_to_z',
            np.array([1., 0., 0.]),
            np.array([0., 0., 1.]),
        ),
        (
            'y_to_x',
            np.array([0., 1., 0.]),
            np.array([1., 0., 0.]),
        ),
        (
            'ppp_to_z',
            np.array([1., 1., 1.]) / np.sqrt(3.),
            np.array([0., 0., 1.]),
        ),
        (
            'unnorm_to_y',
            np.array([3., 0., 4.]),
            np.array([0., 1., 0.]),
        ),
    ]

    os.makedirs(test_data_dir, exist_ok=True)
    path = os.path.join(test_data_dir, 'rotation_matrices.hdf5')

    with h5py.File(path, 'w') as f:
        for name, a, b in test_cases:
            an = a / np.linalg.norm(a)
            bn = b / np.linalg.norm(b)
            R = vrrotvec2mat(vrrotvec(an, bn))
            grp = f.create_group(name)
            grp.create_dataset('a', data=an)
            grp.create_dataset('b', data=bn)
            grp.create_dataset('R', data=R)

    print(f'Saved rotation matrix test data to {path}')
    return None


def save_test_map(map_path, data_path):
    import os
    import h5py
    import uci_tools
    if os.path.exists(map_path):
        answer = input(
            '{0} already exists, and this function will IRREVERSIBLY'
            ' overwrite it. Are you sure you want to do this?'
            '\n(y/n): '
            .format(map_path)
        )
        if answer.lower() not in ('yes', 'y'):
            print('Exited')
            return None
    data = uci_tools.vel_map.load_m12_data_olti(
        data_path,
        snap='600',
        xmax=None,
        zmax=None
    )
    (
        velmap_gas,
        velmap_stars,
        x_edges_gas,
        z_edges_gas,
        x_edges_stars,
        z_edges_stars,
        quadmesh_gas,
        quadmesh_stars,
    ) = uci_tools.vel_map.plot(
        *data,
        display_name='Thelma downsampled',
        snap='600',
        horiz_axis=0,
        vert_axis=2,
        res=100,
        min_gas_sden=0.,
        min_stars_sden=0.,
        save_plot=False
    )
    with h5py.File(map_path, 'w') as f:
        f.create_dataset('velmap_gas', data=velmap_gas)
        f.create_dataset('velmap_stars', data=velmap_stars)
        f.create_dataset('x_edges_gas', data=x_edges_gas)
        f.create_dataset('z_edges_gas', data=z_edges_gas)
        f.create_dataset('x_edges_stars', data=x_edges_stars)
        f.create_dataset('z_edges_stars', data=z_edges_stars)
    print('Finished')
    return None


def save_gbl_data(test_data_dir):
    from uci_tools import config
    import h5py
    import os

    data_dir = config.config.get(f'uci_tools_paths', 'firebox_data_dir')
    direc = os.path.join(
        data_dir,
        'global_sample_data'
    )
    path = os.path.join(
        direc,
        'global_sample_data_snapshot_1200.hdf5'
    )
    with h5py.File(path, 'r') as f_in:
        ids = f_in['galaxyID'][()]
        grp_ids = f_in['groupID'][()]
    is_0 = ids == 0

    if not os.path.isdir(test_data_dir):
        os.makedirs(test_data_dir)
    test_data_path = os.path.join(
        test_data_dir,
        'global_sample_data_snapshot_1200.hdf5'
    )
    with h5py.File(test_data_path, 'w') as f_out:
        f_out.create_dataset('groupID', data=grp_ids[is_0])
        f_out.create_dataset('galaxyID', data=ids[is_0])

    return None
