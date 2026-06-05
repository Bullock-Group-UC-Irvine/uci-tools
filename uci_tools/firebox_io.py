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


def get_grp_id(gal_id):
    '''
    Parameters
    ----------
    gal_id: int
        galaxy of interest

    Returns
    -------
    grp_id: int
        Group ID of the gal. Hosts are -1.
    '''
    df = load_grp_ids()
    return df.loc[gal_id, 'grp_id']

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


def show_gal(gal_id, scaling='std_asinh'):
    from . import config, tools

    scaling_options = ('std_asinh', 'log')
    if scaling not in scaling_options:
        raise ValueError(
            f"scaling must be one of {scaling_options}, got {scaling!r}"
        )

    path = get_gal_path(gal_id)
    fov = get_fov(gal_id)
    half_fov = fov / 2.

    proj_imgs = {}
    with h5py.File(path, 'r') as f:
        proj_imgs['projection_xy'] = np.array(
            f['projection_xy']['band_g']
        )
        proj_imgs['projection_yz'] = np.array(
            f['projection_yz']['band_g']
        )
        proj_imgs['projection_zx'] = np.array(
            f['projection_zx']['band_g']
        )

    octant_img_dir = config.config.get(
        f'{__package__}_paths',
        'octant_img_dir'
    )
    octant_path = find_gal_in_direc(gal_id, octant_img_dir)
    if octant_path != 0:
        with h5py.File(octant_path, 'r') as f:
            for proj in f.keys():
                proj_imgs[proj] = f[proj]['band_g'][:]

    proj_names = list(proj_imgs.keys())
    imgs = [proj_imgs[name] for name in proj_names]

    if scaling == 'std_asinh':
        imgs = [
            tools.std_asinh(
                img[np.newaxis, np.newaxis],
                stretch=1.e-5,
                means=[0.6394],
                stds=[1.2695]
            ).squeeze()
            for img in imgs
        ]

    fig, axs = plt.subplots(4, 3, figsize=(12, 14))
    axs = axs.ravel()
    extent = (-half_fov, half_fov, -half_fov, half_fov)
    for image, proj_name, ax in zip(imgs, proj_names, axs):
        if scaling == 'std_asinh':
            ax.imshow(
                image,
                cmap='gray',
                extent=extent,
            )
        else:
            ax.imshow(
                image,
                cmap='gray',
                interpolation='none',
                norm=mpl.colors.LogNorm(
                    vmin=5.e5,
                    vmax=1.e8
                ),
                extent=extent,
            )
        ax.set_title(proj_name.removeprefix('projection_'))
        ax.set_xlabel('kpc')
        ax.set_ylabel('kpc')
        ax.set_facecolor('k')
        ax.set_aspect('equal')
    for ax in axs[len(imgs):]:
        ax.set_visible(False)
    fig.tight_layout()
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

# Maps the particle_type argument of load_particles to the HDF5 key
# prefix and the partTypes integer used in the ahf bound-filter file.
_PARTICLE_INFO = {
    'gas':     {'prefix': 'gas',      'part_type': 0},
    'stellar': {'prefix': 'stellar',  'part_type': 4},
}


def load_particles(particle_type, obj_path, only_bound=False):
    '''
    Load particles of the given type from a FIREBox HDF5 file without
    FOV filtering. The caller applies the FOV filter after any
    rotation. When only_bound is True, the function looks for the
    bound_particle_filters file in the same directory as obj_path and
    filters to bound particles; if that file does not exist, the
    function returns all particles.

    Parameters
    ----------
    particle_type: str, {'gas', 'stellar'}
        The type of particles to load.
    obj_path: str
        Path to a particles_within_Rvir HDF5 file.
    only_bound: bool, default False
        When True, filter to bound particles using the
        bound_particle_filters file in the same directory.

    Returns
    -------
    coords: np.ndarray, shape (N, 3)
        Particle coordinates in physical kpc.
    vs: np.ndarray, shape (N, 3)
        Particle velocities in km/s.
    ms: np.ndarray, shape (N,)
        Particle masses in units of 1e10 Msun.
    ids: np.ndarray, shape (N,)
        Particle IDs.
    '''
    if particle_type not in _PARTICLE_INFO:
        raise ValueError(
            f'particle_type must be one of '
            f'{list(_PARTICLE_INFO)}, got {particle_type!r}'
        )
    pfx       = _PARTICLE_INFO[particle_type]['prefix']
    part_type = _PARTICLE_INFO[particle_type]['part_type']
    h = 0.6774

    with h5py.File(obj_path, 'r') as f:
        z_snap = f['redshift'][()]
        a_snap = 1. / (1. + z_snap)
        length_conv = a_snap / h

        xs  = f[pfx + '_x'][()]
        ys  = f[pfx + '_y'][()]
        zs  = f[pfx + '_z'][()]
        vxs = f[pfx + '_vx'][()]
        vys = f[pfx + '_vy'][()]
        vzs = f[pfx + '_vz'][()]
        ms_raw = f[pfx + '_mass'][()]
        ids    = f[pfx + '_id'][()]

    if only_bound:
        ahf_name = os.path.basename(obj_path).replace(
            'particles_within_Rvir_', 'bound_particle_filters_',
        )
        ahf_path = os.path.join(os.path.dirname(obj_path), ahf_name)
        if os.path.exists(ahf_path):
            with h5py.File(ahf_path, 'r') as ahf:
                bound_ids = ahf['particleIDs'][
                    ahf['partTypes'][()] == part_type
                ]
            _, idx, _ = np.intersect1d(
                ids,
                bound_ids,
                assume_unique=False,
                return_indices=True,
            )
            xs,  ys,  zs  = xs[idx],  ys[idx],  zs[idx]
            vxs, vys, vzs = vxs[idx], vys[idx], vzs[idx]
            ms_raw = ms_raw[idx]
            ids    = ids[idx]

    coords = np.column_stack([xs, ys, zs]) * length_conv
    vs     = np.column_stack([vxs, vys, vzs])
    ms     = ms_raw / h
    return coords, vs, ms, ids


def get_bound_particles(gal_id):
    from . import config
    import os
    import h5py

    super_dir = config.config.get(f'{__package__}_paths', 'firebox_data_dir')
    firebox_snap = config.config.get(f'{__package__}_paths', 'firebox_snap')
    path = os.path.join(
        super_dir,
        firebox_snap,
        f"bound_particle_filters_object_{str(gal_id)}.hdf5"
    )
    with h5py.File(path, 'r') as f:
        particle_ids = np.array(f['particleIDs'], int)
    return particle_ids
