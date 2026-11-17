'''
Derive per-galaxy properties from FIREBox particle data.

This is the Python companion to the Julia ProcessFIREBox module. Each public
function turns raw particle files into a summary property, for one galaxy or
for the whole sample, the way get_avg_sfrs and get_all_avg_sfrs do there.

All masses here are physical solar masses, the scale firebox_io.load_particles
returns after it applies the 1/h little-h correction.
'''
import os
import multiprocessing

import numpy as np
import pandas as pd

from . import config
from . import firebox_io


def half_mass_radius(radii, masses):
    '''
    Find the 3D radius enclosing half of a mass distribution.

    Parameters
    ----------
    radii: np.ndarray, shape (N,)
        Particle distances from the galaxy center.
    masses: np.ndarray, shape (N,)
        Particle masses. A constant scale on them cancels, so their units do
        not matter here.

    Returns
    -------
    radius: float
        The radius at which the cumulative mass first reaches half the total,
        in the units of radii.
    '''
    order = np.argsort(radii)
    cumulative = np.cumsum(masses[order])
    i = np.searchsorted(cumulative, 0.5 * cumulative[-1])
    return radii[order][i]


def enclosed_mass(radii, masses, radius):
    '''
    Sum the mass of the particles that sit inside a radius.

    Parameters
    ----------
    radii: np.ndarray, shape (N,)
        Particle distances from the galaxy center.
    masses: np.ndarray, shape (N,)
        Particle masses, in the units the caller wants the sum in.
    radius: float
        The enclosing radius, in the units of radii.

    Returns
    -------
    mass: float
        Total mass of the particles with radii < radius.
    '''
    return masses[radii < radius].sum()


def _radii(coords):
    '''
    Return each particle's distance from the origin, where
    firebox_io.load_particles has already centered the galaxy.

    Parameters
    ----------
    coords: np.ndarray, shape (N, 3)
        Particle coordinates in physical kpc.

    Returns
    -------
    radii: np.ndarray, shape (N,)
        Distances from the galaxy center, in physical kpc.
    '''
    return np.linalg.norm(coords, axis=1)


def _load_radii_masses(particle_type, obj_path):
    '''
    Load one particle type's distances from the galaxy center and its masses,
    in physical units. Return empty arrays when the type is absent from the
    file, so a species a galaxy lacks (for example a gas-free galaxy)
    contributes zero enclosed mass instead of raising.

    Parameters
    ----------
    particle_type: str
        One of the types firebox_io.load_particles accepts.
    obj_path: str
        Path to the galaxy's within-Rvir particle file.

    Returns
    -------
    radii: np.ndarray, shape (N,)
        Distances from the galaxy center in physical kpc, empty when the type's
        arrays are not in the file.
    masses: np.ndarray, shape (N,)
        Particle masses in physical Msun, empty when the type's arrays are not
        in the file.
    '''
    try:
        coords, _, masses, _ = firebox_io.load_particles(
            particle_type,
            obj_path,
        )
    except KeyError:
        return np.empty(0), np.empty(0)
    return _radii(coords), masses * 1.e10


def mtot_within_rhalfstar(gal_id):
    '''
    Compute the total mass within one galaxy's half-stellar-mass radius.

    The half-stellar-mass radius R_half,* is the 3D radius enclosing half of
    the galaxy's stellar mass, measured from the particles, so it is robust and
    projection-free. The stellar mass inside it is 0.5 * Mstar by construction,
    but the enclosed gas and dark matter vary from galaxy to galaxy, so the
    total (stars + gas + dark matter) is a non-degenerate central mass scale.
    R_half,* and the enclosed masses count every particle in the within-Rvir
    file, including the accreted stellar halo and satellites.

    Parameters
    ----------
    gal_id: int
        The galaxy to process.

    Returns
    -------
    record: dict or None
        The keys galaxyID, r_half_star (physical kpc), mstar_rhalfstar,
        mgas_rhalfstar, mdm_rhalfstar, and mtot_rhalfstar (physical Msun). None
        when the galaxy has no particle file or no stellar particles.
    '''
    obj_path = firebox_io.get_particle_path(gal_id)
    if not os.path.isfile(obj_path):
        return None

    s_radii, s_masses = _load_radii_masses('stellar', obj_path)
    if len(s_masses) == 0:
        print(
            '  skipping object_{0}: no stellar particles'.format(gal_id),
            flush=True,
        )
        return None
    r_half = half_mass_radius(s_radii, s_masses)

    g_radii, g_masses = _load_radii_masses('gas', obj_path)
    d_radii, d_masses = _load_radii_masses('dm', obj_path)

    m_star = enclosed_mass(s_radii, s_masses, r_half)
    m_gas = enclosed_mass(g_radii, g_masses, r_half)
    m_dm = enclosed_mass(d_radii, d_masses, r_half)
    return {
        'galaxyID': int(gal_id),
        'r_half_star': r_half,
        'mstar_rhalfstar': m_star,
        'mgas_rhalfstar': m_gas,
        'mdm_rhalfstar': m_dm,
        'mtot_rhalfstar': m_star + m_gas + m_dm,
    }


def get_all_mtot_rhalfstar(save=False, nworkers=None):
    '''
    Compute the total mass within the half-stellar-mass radius for every galaxy
    in the FIREBox sample.

    Parameters
    ----------
    save: bool, default False
        When True, write the result to mtot_rhalfstar.csv in the uci_tools
        project_data_dir.
    nworkers: int or None, default None
        Number of worker processes. None uses every core the job may run on
        (os.sched_getaffinity), which under SLURM is the set the allocation
        granted. The work is independent per galaxy, and its cost is dominated
        by reading each galaxy's particle arrays from disk, so parallel workers
        cut the wall time substantially.

    Returns
    -------
    df: pd.DataFrame
        One row per galaxy that has a particle file, sorted by galaxyID, with
        the columns galaxyID, r_half_star, mstar_rhalfstar, mgas_rhalfstar,
        mdm_rhalfstar, and mtot_rhalfstar.
    '''
    gal_ids = [int(g) for g in firebox_io.load_grp_ids().index]
    if nworkers is None:
        nworkers = len(os.sched_getaffinity(0))
    n = len(gal_ids)
    print(
        'Computing mtot_rhalfstar for {0} galaxies on {1} workers.'
        .format(n, nworkers)
    )

    records = []
    done = 0
    with multiprocessing.Pool(nworkers) as pool:
        results = pool.imap_unordered(
            mtot_within_rhalfstar,
            gal_ids,
        )
        for record in results:
            done += 1
            if record is None:
                continue
            records.append(record)
            print(
                '  {0}/{1}  object_{2}: R_half,* = {3:.2f} kpc, '
                'Mtot(<R_half,*) = {4:.3e} Msun'.format(
                    done,
                    n,
                    record['galaxyID'],
                    record['r_half_star'],
                    record['mtot_rhalfstar'],
                ),
                flush=True,
            )

    columns = [
        'galaxyID',
        'r_half_star',
        'mstar_rhalfstar',
        'mgas_rhalfstar',
        'mdm_rhalfstar',
        'mtot_rhalfstar',
    ]
    df = pd.DataFrame.from_records(records)[columns]
    df = df.sort_values('galaxyID').reset_index(drop=True)

    if save:
        out_dir = config.config.get(
            f'{__package__}_paths',
            'project_data_dir',
        )
        out_path = os.path.join(out_dir, 'mtot_rhalfstar.csv')
        df.to_csv(out_path, index=False, float_format='%.6e')
        print('wrote {0} ({1} galaxies)'.format(out_path, len(df)))

    return df
