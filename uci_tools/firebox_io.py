import os
import h5py
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

def load_grp_ids():
    from . import config
    data_dir = config.config.get(f'{__package__}_paths', 'firebox_data_dir')
    fname = os.path.join(
        data_dir,
        'global_sample_data',
        'global_sample_data_snapshot_1200.hdf5'
    )
    d = {}
    with h5py.File(fname, 'r') as f:
        d['grp_id'] = f['groupID'][()].astype(int)
        ids = f['galaxyID'][()].astype(int)
    df = pd.DataFrame(d, index=ids)
    return df

def find_gal_in_direc(gal_id, direc):
    all_files = np.array(
        [f 
         for f in os.listdir(direc) 
         if os.path.isfile(os.path.join(direc, f))]
    )
    mask = ['object_' + str(gal_id) + '_' in fname for fname in all_files]
    if np.sum(mask) > 1:
        raise(Exception('ID matched more than one file.'))
    elif np.sum(mask) == 1:
        fname = all_files[mask][0]
        path = os.path.join(direc, fname)
        return path
    else:
        return 0

def get_gal_path(gal_id):
    '''
    Get the path to the image of the given galaxy.
    '''
    from . import config
    host_direc = config.config.get(f'{__package__}_paths', 'host_image_dir')
    sat_direc = config.config.get(f'{__package__}_paths', 'sat_image_dir')
    path = find_gal_in_direc(gal_id, host_direc)
    if path == 0:
        path = find_gal_in_direc(gal_id, sat_direc)
    if path == 0:
        raise(Exception('Galaxy not found.'))
    return path

def show_gal(gal_id):
    path = get_gal_path(gal_id)

    with h5py.File(path, 'r') as f:
        image_xy = np.array(f['projection_xy']['band_g'])
        image_yz = np.array(f['projection_yz']['band_g'])
        image_zx = np.array(f['projection_zx']['band_g'])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.01)
    for image, ax in zip([image_xy, image_yz, image_zx], axs):
        ax.imshow(
            image,
            cmap='gray', 
            interpolation='none',
            norm=mpl.colors.LogNorm(
                    vmin=5.e5,
                    vmax=1.e8
                ),
            #origin='lower'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('k')
        ax.set_aspect('equal')
    plt.show()

    return None

def get_fov(gal_id):
    import h5py
    import pandas as pd
    import numpy as np

    from . import config

    grp_id = load_grp_ids().loc[gal_id, 'grp_id']
    if grp_id == -1:
        # Host
        path = config.config.get(f'{__package__}_paths', 'host_2d_shapes')
    else:
        # Satellite
        path = config.config.get(f'{__package__}_paths', 'sat_2d_shapes')
    df = pd.read_csv(path, index_col='galaxyID')
    fov = df.loc[gal_id, 'FOV']
    
    if np.all(fov == fov.iloc[0]):
        fov = fov.iloc[0]
    else:
        raise ValueError(
            f'{gal_id} has mutliple values for field of view in {path}'
        )

    return fov

def load_particle(
        particle_str,
        f,
        gal_id,
        horiz_axis,
        vert_axis,
        only_bound=True):
    import numpy as np

    ys = f[particle_str + '_y'][()]
    zs = f[particle_str + '_z'][()]
    xs = f[particle_str + '_x'][()]
    vxs = f[particle_str + '_vx'][()]
    vys = f[particle_str + '_vy'][()]
    vzs = f[particle_str + '_vz'][()]
    ms = f[particle_str + '_mass'][()]
    ids = f[particle_str + '_id'][()]

    coords = np.array([xs, ys, zs]).T
    vs = np.array([vxs, vys, vzs]).T
    
    proj_coords = coords[:, [horiz_axis, vert_axis]]
    dists_2d = np.linalg.norm(proj_coords, axis=1)
    fov = get_fov(gal_id)
    in_fov = np.abs(dists_2d) <= fov / 2.
    
    coords = coords[in_fov]
    vs = vs[in_fov]
    ms = ms[in_fov]
    ids = ids[in_fov]

    if only_bound:
        grp_id = load_grp_ids().loc[gal_id, 'grp_id']

        if grp_id != -1:
            # If it's a satellite, get only bound particles
            bound_ids = get_bound_particles(gal_id)
            (
                intersect1d,
                indices,
                comm2
            ) = np.intersect1d(
                ids,
                bound_ids,
                assume_unique=False,
                return_indices=True
            )
            print(intersect1d)
            coords = coords[indices]
            vs = vs[indices]
            ms = ms[indices]
            ids = ids[indices]

    return coords, vs, ms, ids, fov

def get_bound_particles(gal_id):
    from . import config
    import os
    import h5py

    super_dir = config.config.get(f'{__package__}_paths', 'firebox_data_dir')
    path = os.path.join(
        super_dir,
        'objects_1200',
        f"bound_particle_filters_object_{str(gal_id)}.hdf5"
    )
    with h5py.File(path, 'r') as f:
        particle_ids = np.array(f['particleIDs'], int)
    return particle_ids
