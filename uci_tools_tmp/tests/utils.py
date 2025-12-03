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
