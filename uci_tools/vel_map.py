def get_m12_path_olti(sim_name, host_idx, snap):
    '''
    Generate the path to simulation files in Olti's directory.

    Parameters
    ----------
    sim_name: {
        'm12b_res7100',
        'm12c_res7100',
        'm12_elvis_RomeoJuliet_res3500',
        'm12_elvis_RomulusRemus_res4000',
        'm12_elvis_ThelmaLouise_res4000',
        'm12f_res7100',
        'm12i_res7100',
        'm12m_res7100',
        'm12r_res7100'
    }
        Name of the simulation to load.
    host_idx: int
        The index of the host to analyze in the given simulation. For Latte
        runs, this should always be 0. For Elvis pairs, the index will either
        be 0 or 1 depending on which of the the pair the user wants to analyze.
    snap: str
        The snapshot number to load. The snapshot number should be in
        string format with three digits.

    Returns
    -------
    path: str
        The path to the file being analyzed.
    '''
    import os
    path = os.path.join(
        '/DFS-L/DATA/cosmo/grenache/omyrtaj/analysis_data/metaldiff/',
        sim_name,
        'id_jnet_jzjc_jjc_'
            + snap
            + '_host'
            + str(host_idx)
            + '_20kpc_rockstar_centers_metaldiff.hdf5'
    )
    return path


def load_m12_data_olti(sim_path, snap, xmax=None, zmax=None):
    '''
    Load Olti's data for use with `uci_tools.vel_map.plot`. You can also
    use this with your own data at `sim_path` as long as it's an hdf5 file with
    the following `h5py.Dataset`s:
        'gas_coord_unroated'
        'gas_vel_unrotated'
        'jnet_gas'
        'gas_temp'
        'mass_gas'
        'star_coord_unrotated'
        'star_vel_unrotated',
        'sft_Gyr',
        'jnet_young_star'

    Parameters
    ----------
    sim_path: str 
        The path to the simulation the user wants to analyze. The user could
        use uci_tools.vel_map.get_m12_path_olti to easily generate a path
        that leads to Olti's files, or they could supply their own path.
        Another option would be for the user to write their own `get_m12_path`
        method in uci_tools.vel_map and make a pull request so everyone
        has it.
    snap: str
        The snapshot number corresponding to `sim_path`. It should be in string
        format with three digits. The code uses this to determine the scale
        factor and thereby physical distances.
    xmax: float, default None
        The absolute value of the rotated x-axis distance from the host center
        that a particle must be at or below for the code to include it.
    zmax: float, default None
        The absolute value of the rotated z-axis distance from the host center
        that a particle must be at or below for the code to include it.

    Returns
    -------
    pos_star: np.ndarray, shape (N_stars, 3)
        The centered, rotated position vectors of the
        simulation's star particles in Cartesian coordinates in physical kpc.
        The function rotated them
        so the the x- and y-axes are in the plane of the disc by aligning 
        the z-axis 
        with the net angular momentum of the young stars.
    vel_star: np.ndarray, shape (N_stars, 3)
        The rotated velocity vectors of the star particle in Cartesian
        coordinates
        relative to
        the host center. The function rotated them
        so the the x- and y-axes are in the plane of the disc by aligning 
        the z-axis 
        with the net angular momentum of the cold gas (T <= 1e4 K)
    mass_star: np.ndarray, shape (N_stars,)
        Masses of the star particles in units of 1e10 M_sun
    pos_gas: np.ndarray, shape (N_gas, 3)
        The centered, rotated position vectors of the
        simulation's gas particles in Cartesian coordinates in physical kpc.
        The function rotated them
        so the the x- and y-axes are in the plane of the disc by aligning 
        the z-axis 
        with the net angular momentum of the cold gas 
        (T <= 1e4 K)
    vel_gas: np.ndarray, shape (N_gas, 3)
        The rotated velocity vectors of the gas particle in Cartesian
        coordinates
        relative to
        the host center. The function rotated them
        so the the x- and y-axes are in the plane of the disc by aligning 
        the z-axis 
        with the net angular momentum of the cold gas (T <= 1e4 K)
    mass_gas: np.ndarray, shape (N_stars,)
        Masses of the gas particles in units of 1e10 M_sun
    '''
    import h5py
    import numpy as np
    from . import config 
    from . import rotate_galaxy

    if xmax is None:
        xmax = np.inf
    if zmax is None:
        zmax = np.inf

    snapshot_times = np.loadtxt(
        config.config['uci_tools_paths']['snap_times']
    )
    time = float(snapshot_times[int(snap)][3])
    lbt = np.abs(time - 13.8)
    a = float(snapshot_times[int(snap)][1])

    data = {}
    with h5py.File(sim_path, 'r') as f:
        host_center = np.array(f['host_center'])
        host_vel = np.array(f['host_velocity'])

        # Load gas data
        pos_gas = a * (
            np.array(f['gas_coord_unrotated']) - host_center
        )
        vel_gas = np.array(f['gas_vel_unrotated']) - host_vel
        jnet_gas = np.array(f['jnet_gas'])
        temp = np.array(f['gas_temp'])
        mass_gas = np.array(f['mass_gas'])
        
        # Load star data
        pos_star = a * (
            np.array(f['star_coord_unrotated']) - host_center
        )
        vel_star = np.array(f['star_vel_unrotated']) - host_vel
        sft = np.array(f['sft_Gyr'])
        jnet_star = np.array(f['jnet_young_star'])
        mass_star = np.array(f['mass'])

    #**************************************************************************
    # Gas
    #**************************************************************************
    is_cool = temp < 1e4
    temp = temp[is_cool]
    pos_gas = pos_gas[is_cool]
    vel_gas = vel_gas[is_cool]
    mass_gas = mass_gas[is_cool]

    jnet_gas = rotate_galaxy.calculate_ang_mom(mass_gas, pos_gas, vel_gas)

    # Rotation matrix
    r_matrix_gas = rotate_galaxy.cal_rotation_matrix(
        jnet_gas,
        np.array((0.0, 0.0, 1.0))
    )
    pos_gas = rotate_galaxy.rotate(pos_gas, r_matrix_gas)
    vel_gas = rotate_galaxy.rotate(vel_gas, r_matrix_gas)

    gas_in_x = np.abs(pos_gas[:, 0]) <= xmax
    gas_in_z = np.abs(pos_gas[:, 2]) <= zmax

    v_gas = np.linalg.norm(vel_gas, axis=1)
    v_max = 220
    gas_below_vmax = v_gas <= v_max

    temp = temp[gas_below_vmax & gas_in_x & gas_in_z]
    pos_gas = pos_gas[gas_below_vmax & gas_in_x & gas_in_z]
    vel_gas = vel_gas[gas_below_vmax & gas_in_x & gas_in_z]
    mass_gas = mass_gas[gas_below_vmax & gas_in_x & gas_in_z]

    #**************************************************************************
    # Stars
    #**************************************************************************
    young_mask = (sft <= (lbt + .5))
    pos_star = pos_star[young_mask]
    vel_star = vel_star[young_mask]
    mass_star = mass_star[young_mask]

    # Rotation matrix
    r_matrix_star = rotate_galaxy.cal_rotation_matrix(
        jnet_star,
        np.array((0.0, 0.0, 1.0))
    )
    pos_star = rotate_galaxy.rotate(pos_star, r_matrix_star)
    vel_star = rotate_galaxy.rotate(vel_star, r_matrix_star)

    stars_in_x = np.abs(pos_star[:, 0]) <= xmax
    stars_in_z = np.abs(pos_star[:, 2]) <= zmax

    v_star = np.linalg.norm(vel_star, axis=1)
    stars_below_vmax = v_star <= v_max
    
    pos_star = pos_star[stars_below_vmax & stars_in_x & stars_in_z]
    vel_star = vel_star[stars_below_vmax & stars_in_x & stars_in_z]
    mass_star = mass_star[stars_below_vmax & stars_in_x & stars_in_z]
    #**************************************************************************

    mass_star /= 1.e10
    mass_gas /= 1.e10
    
    return pos_star, vel_star, mass_star, pos_gas, vel_gas, mass_gas


def calc_vmap(coords, vs, ms, horiz_axis, vert_axis, res, min_cden):
    '''
    Calculate the velocity map for the particles that the user provides.

    Parameters
    ----------
    coords: np.ndarray, shape (N, 3)
        Cartesian position vectors relative to the host center, in physical
        kpc, of the
        particles whose velocities the user wants to map.
        The resulting velocity map takes the line of sight to be the axis that
        is perpendicular to the `horiz_axis` and `vert_axis`. For example, if
        the user provides rotated vectors with their z-axis
        aligned with the galaxy's net angular momentum and specifies
        `horiz_axis=0, vert_axis=2`, this is analagous to
        looking into the disc edge-on.
    vs: np.ndarray, shape (N, 3)
        Cartesian velocity vectors relative to the host center, in physical
        km/s, of the particles whose
        velocities the user wants to map.
        If
        the user provides rotated vectors with their z-axis
        aligned with the galaxy's net angular momentum and specifies
        `horiz_axis=0, vert_axis=2`, this is analagous to
        looking into the disc edge-on.
    ms: np.ndarray, shape (N,)
        The masses of the particles, in units of 1e10 M_sun, whose
        velocities
        the user wants to map.
    horiz_axis: int, default 0
        The index that would appear horizontally if the user printed the
        results. Note that this is the 1 axis (columns) of the resulting array.
    vert_axis: int, default 2
        The index that would appear vertically if the user printed the
        results. Note that this is the 0 axis (rows) of the resulting array.
    res: int
        Resolution: the number of pixels (i.e. bins) in each axis of the
        velocity map.
    min_cden: float
        The minimum column density in M_sun / pc^2 of particles for a pixel
        in the velocity map to be given a numerical value. Otherwise the pixel
        is np.nan.

    Returns
    -------
    vmap, np.ndarray, shape (res, res)
        The velocity map data for analysis or for use with 
        `matplotlib.axes.Axis.pcolormesh`. Horizontal data is along the 1 axis.
        Vertical data is along the 0 axis. The user can directly input this
        into
        `pcolormesh`.
    x_edges, np.ndarray, shape (res,)
        The locations of the edges of the velocity map pixels in kpc along
        the
        horizontal axis
    z_edges, np.ndarray, shape (res,)
        The locations of the edges of the velocity map pixels in kpc along
        the
        vertical axis
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    axes = [0, 1, 2]
    # Determine which axis the user did not specify as a projection axis 
    los_axis = np.setdiff1d(axes, [horiz_axis, vert_axis])
    if len(los_axis) > 1:
        # There should only be one line-of-sight axis
        raise ValueError('Something is wrong with the axis specifications')
    los_axis = los_axis[0]

    # Only the `los_axis`-axis velocity is necessary.
    v_y = vs[:, los_axis]  # Use for colormap
    # Need to subract off the average velocity. Gas may be moving differently
    # from the halo.
    mass_weighted_v = np.average(v_y, weights=ms)
    avg_v = v_y.mean()
    v_y -= mass_weighted_v

    # The function looks down the `los_axis`-axis and so does not use its
    # positional
    # information.
    x = coords[:, horiz_axis]
    #y = coords[:, los_axis]
    z = coords[:, vert_axis]

    # Create 2D histogram
    hist, x_edges, z_edges = np.histogram2d(
        x,
        z,
        bins=res
    )
    # Need to reverse z_edges so that z_edges[0] is the highest z. This is what
    # matplotlib.pyplot.imshow expects, and it makes sense when one considers
    # that when printing the mesh data, high z should be on the top of the
    # matrix.
    z_edges = z_edges[::-1]
    hist = hist[:, ::-1]

    hist += 1  # Avoid log(0)

    #**********************************************************************
    # Bin the v_y values and create a colormap based on the average 
    # v_y in each bin

    # Get indices of the x and z locations into which each particle falls
    x_bin_indices = np.digitize(x, x_edges) - 1
    z_bin_indices = np.digitize(z, z_edges) - 1
    v_y_colormap = np.zeros_like(hist)
    cden_map = np.zeros_like(hist)
    count_map = np.zeros_like(hist)

    bin_area = (
        (x_edges.max() - x_edges.min()) / (len(x_edges) - 1.)
        * (z_edges.max() - z_edges.min()) / (len(z_edges) - 1.)
    )
    for i in range(len(x)):
        # For single particle, add its velocity to its position in the
        # velocity map, add a 1 to its position in the
        # count map, and add it smass to the mass map.
        if (
                0 <= x_bin_indices[i] < res 
                and 0 <= z_bin_indices[i] < res):
            # We use `< res`, not `<= res` because np.digitize with right=False
            # assigns x to a bin beyond the histogram when x == x_edges.max().
            # i.e. Without this, we would have some `i` that are beyond
            # `len(x_bin_indices) - 1`
            v_y_colormap[
                x_bin_indices[i],
                z_bin_indices[i]
            ] += v_y[i]
            count_map[x_bin_indices[i], z_bin_indices[i]] += 1
            # Adding to the given pixel's column brightness in units of
            # 1e10 M_sun / kpc^2
            cden_map[x_bin_indices[i], z_bin_indices[i]] += (
                ms[i] / bin_area
            )
    
    # Convert column density map from 1e10 M_sun / kpc^2 to M_sun / pc^2
    cden_map *= 1.e10 / 1.e3 / 1.e3

    inspect_cdens = False
    if inspect_cdens:
        masses = cden_map.flatten()
        bin_start = np.log10(np.sort(list(set(masses)))[1])
        bins = np.logspace(bin_start, np.log10(masses.max()), 50)
        plt.hist(masses, bins=bins)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    # Avoid division by zero:
    count_map[count_map == 0] = 1
    # Finally, calculating the avg v_y in each bin:
    v_y_colormap /= count_map

    # Apply the mask to keep bins with at least a `min_cden` column density
    mask = cden_map >= min_cden
    vmap = np.where(mask, v_y_colormap, np.nan)

    # The `H` output of `np.histogram2d` has x data along the 0 axis and y data 
    # along the 1 axis.
    # On one hand, that makes sense; when we specify coordinates, we tend to
    # specify x first. E.g. (x, y). However, if one prints out `H`, visually,
    # one might expect the x-axis to run horizontally along `H` and the y-axis 
    # to
    # run vertically along `H`. This is not the case. Additionally, 
    # `pcolormesh`
    # expects the column number as x and the row number as y 
    # Therefore, we must provide the transpose of `H` to
    # `pcolormesh`.
    vmap = vmap.T

    return vmap, x_edges, z_edges


def plot(
        pos_star,
        vel_star,
        mass_star,
        pos_gas,
        vel_gas,
        mass_gas,
        display_name,
        snap,
        horiz_axis=0,
        vert_axis=2,
        res=100,
        min_gas_cden=14.,
        min_stars_cden=40.,
        save_plot=False,
        show_plot=True):
    '''
    Plot the v_y velocity map of gas and young stars for a given simulation and
    return the data for those two maps.
     
    The grid orientation of the returned maps 
    follows the standard matrix convention specified in the 
    `matplotlib.axes.Axes.pcolormesh` documentation; They have shape 
    (nrows, ncolumns) with the column number as X and the row number as Y. 
    
    Note
    that this orientation is the transpose of the `H` output of 
    `np.histogram2d`,
    which has X data along the 0 axis and Y data 
    along the 1 axis.
    If one were to print out `H`, 
    visually,
    one might expect the X-axis to run horizontally along `H` and the 
    Y-axis to
    run vertically along `H`. However, the opposite is true. Additionally, 
    `matplotlib.axes.Axes.pcolormesh`
    plots the inputted `C` mesh arry with the column number as X and the row 
    number as Y.
    Therefore, we must provide the transpose of `H` to
    `pcolormesh`.

    Parameters
    ----------
    pos_star: np.ndarray, shape (N_stars, 3)
        Cartesian position vectors relative to the host center, in physical
        kpc, of the
        star particles whose velocities the user wants to map.
        The resulting velocity map takes the line of sight to be the axis that
        is perpendicular to the `horiz_axis` and `vert_axis`. For example, if
        the user provides rotated vectors with their z-axis
        aligned with the galaxy's net angular momentum and specifies
        `horiz_axis=0, vert_axis=2`, this is analagous to
        looking into the disc edge-on.
    vel_star: np.ndarray, shape (N_stars, 3)
        Cartesian velocity vectors relative to the host center, in physical
        km/s, of the star particles whose
        velocities the user wants to map.
        If
        the user provides rotated vectors with their z-axis
        aligned with the galaxy's net angular momentum and specifies
        `horiz_axis=0, vert_axis=2`, this is analagous to
        looking into the disc edge-on.
    mass_star: np.ndarray, shape (N_star,)
        The masses of the star particles, in units of 1e10 M_sun, whose
        velocities
        the user wants to map.
    pos_gas: np.ndarray, shape (N_gas, 3)
        Cartesian position vectors relative to the host center, in physical
        kpc, of the
        gas particles whose velocities the user wants to map.
        The resulting velocity map takes the line of sight to be the axis that
        is perpendicular to the `horiz_axis` and `vert_axis`. For example, if
        the user provides rotated vectors with their z-axis
        aligned with the galaxy's net angular momentum and specifies
        `horiz_axis=0, vert_axis=2`, this is analagous to
        looking into the disc edge-on.
    vel_gas: np.ndarray, shape (N_gas, 3)
        Cartesian velocity vectors relative to the host center, in physical
        km/s, of the gas particles whose
        velocities the user wants to map.
        If
        the user provides rotated vectors with their z-axis
        aligned with the galaxy's net angular momentum and specifies
        `horiz_axis=0, vert_axis=2`, this is analagous to
        looking into the disc edge-on.
    mass_gas: np.ndarray, shape (N_star,)
        The masses of the gas particles, in units of 1e10 M_sun, whose
        velocities
        the user wants to map.
    display_name: str 
        Simulation name to show in the plot.
    snap: str
        The snapshot number corresponding to the data. It should be in string
        format with three digits. The code uses this to
        display the correct look-back time in the plot.
    horiz_axis: int, default 0
        The index of the axis to show horizontally on the plot. The default is 
        the 0
        or x-axis.
    vert_axis: int, default 2
        The index of the axis to show vertically on the plot. The default is
        the 2 or z-axis.
    res: int, default 100
        Resolution: the number of pixels (i.e. bins) in each axis.
    min_gas_cden: float, default 14.
        The minimum column density in M_sun / pc^2 of gas particles for a pixel
        in the velocity map to be given a numerical value. Otherwise the pixel
        is np.nan.
    min_stars_cden: float, default 40.
        The minimum column density in M_sun / pc^2 of star particles for a
        pixel
        in the velocity map to be given a numerical value. Otherwise the pixel
        is np.nan.
    save_plot: bool, default True
        Whether to save the plot to disk. If True, the code will save the plot
        in the `project_data_dir` directory specified in the user's 
        config.ini file in
        their home directory. 
    show_plot: bool, default True
        Whether to display the velocity map.

    Returns
    -------
    velmap_gas: np.ndarray
        The gas velocity colormap data for use with 
        `matplotlib.axes.Axis.pcolormesh`. X data is along the 1 axis. Z data
        is along the 0 axis. The user can directly input this into
        `pcolormesh`.
    velmap_star: np.ndarray
        The young-star velocity colormap data for use with 
        `matplotlib.axes.Axis.pcolormesh`. X data is along the 1 axis. Z data
        is along the 0 axis. The user can directly input this into
        `pcolormesh`.
    x_edges_gas: np.ndarray, shape (res,)
        The locations of the edges of the gas velocity map pixels in kpc along
        the
        horizontal axis
    z_edges_gas: np.ndaray, shape (res,)
        The locations of the edges of the gas velocity map pixels in kpc
        along the
        vertical axis
    x_edges_star: np.ndarray, shape (res,)
        The locations of the edges of the stellar velocity map pixels in kpc
        along
        the
        horizontal axis
    z_edges_star: np.ndarray, shape (res,)
        The locations of the edges of the stellar velocity map pixels in kpc
        along
        the
        vertical axis
    quadmesh_gas: matplotlib.collections.QuadMesh
        The output of the plt.pcolormesh that creates the velocity map visual
        for the gas. The user can use this to replicate the exact visual
        at a later time.
    quadmesh_star: matplotlib.collections.QuadMesh
        The output of the plt.pcolormesh that creates the velocity map visual
        for the stars. The user can use this to replicate the exact visual
        at a later time.
    '''

    import os
    import h5py
    import numpy as np

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    import astropy.cosmology as cosmo
    import astropy

    from . import config

    snapshot_times = np.loadtxt(
        config.config['uci_tools_paths']['snap_times']
    )
    time = float(snapshot_times[int(snap)][3])
    lbt = np.abs(time - 13.8)

    velmap_gas, x_edges_gas, z_edges_gas = calc_vmap(
        pos_gas,
        vel_gas,
        mass_gas,
        horiz_axis,
        vert_axis,
        res,
        min_gas_cden,
    )
    velmap_star, x_edges_star, z_edges_star = calc_vmap(
        pos_star,
        vel_star,
        mass_star,
        horiz_axis,
        vert_axis,
        res,
        min_stars_cden,
    )

    # Set up the figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Get global vmin and vmax for the colormap based on both gas and stars
    #vmin = min(np.nanmin(velmap_gas), np.nanmin(velmap_star))
    vmax = max(np.nanmax(velmap_gas), np.nanmax(velmap_star))
    vmin = -1*vmax

    # Plot gas with colormap based on v_y_gas
    quadmesh_gas = ax[0].pcolormesh(
        x_edges_gas,
        z_edges_gas,
        velmap_gas,
        cmap=plt.cm.seismic_r,
        vmin=vmin,
        vmax=vmax
    )

    # Plot stars with colormap based on v_y_star
    quadmesh_star = ax[1].pcolormesh(
        x_edges_star,
        z_edges_star,
        velmap_star,
        cmap=plt.cm.seismic_r,
        vmin=vmin,
        vmax=vmax
    )

    if show_plot or save_plot:
        # Add colorbars for both plots
        fig.colorbar(
            quadmesh_gas,
            ax=ax[0],
            label=r'Gas LOS Velocity [kms$^{-1}]$'
        )
        fig.colorbar(
            quadmesh_star,
            ax=ax[1],
            label=r'Star LOS Velocity [kms$^{-1}]$'
        )

        # Add labels and text
        ax[0].text(
            0.9,
            0.9,
            'Gas',
            transform=ax[0].transAxes,
            fontsize=16,
            color='k',
            ha='right'
        )
        ax[1].text(
            0.9,
            0.9,
            'Stars',
            transform=ax[1].transAxes,
            fontsize=16,
            color='k',
            ha='right'
        )

        ax[0].text(
            0.1,
            0.9,
            '{0}'.format(display_name),
            transform=ax[0].transAxes,
            color='k',
            fontsize=16
        )
        ax[0].text(
            0.1,
            0.85,
            'LBT = ' + str(np.round(lbt, 2)) + ' Gyr',
            transform=ax[0].transAxes,
            color='k',
            fontsize=14
        )

        # Axis labels
        axis_labels = ['$x$', '$y$', '$z$']
        ax[0].set_ylabel(
            '{0} [kpc]'.format(axis_labels[vert_axis]),
            fontsize=16
        )
        for i in range(len(ax)):
            ax[i].set_xlabel(
                '{0} [kpc]'.format(axis_labels[horiz_axis]),
                fontsize=16
            )
            ax[i].tick_params(
                axis='x',
                direction='in',
                pad=10,
                which='both',
                top=True,
                bottom=True,
                color='k',
                length = 6,
                width = 1.3
            )
            ax[i].tick_params(
                axis='y',
                direction='in',
                pad=10,
                which='both',
                left=True,
                right=True,
                color='k',
                length = 6,
                width = 1.3
            )

        # Tight layout and spacing
        plt.tight_layout()

        if save_plot:
            plt.savefig(os.path.join(
                config.config['uci_tools_paths']['project_data_dir'], 
                'vel_map_{0}_snap{1}.png'.format(display_name.lower(), snap)
            ))
        if show_plot:
            plt.show()
    else:
        plt.close()

    return (
        velmap_gas,
        velmap_star,
        x_edges_gas,
        z_edges_gas,
        x_edges_star,
        z_edges_star,
        quadmesh_gas,
        quadmesh_star,
    )


def firebox_vmap(gal_id, res, min_cden=14., queue=None):
    '''
    Create bound-gas velocity maps for all 11 projections of a FIREBox
    galaxy: the 3 standard axis-aligned projections (xy, yz, zx) plus
    the 8 octant body-diagonal projections (ppp, ppm, ..., mmm).

    Reads particle data from firebox_data_dir/firebox_snap, with both
    paths set in the uci_tools_paths section of the config.
    The FOV for each map matches the FOV in Courtney's mock image for
    that galaxy. Returns None if the particle file does not exist or
    if the galaxy has no bound gas particles.

    All 11 projection groups are written to a single output file in
    project_data_dir/vmaps_res{res}_min_cden{min_cden}/, so
    image_loader.jl can read standard and octant projections without
    any changes.

    All viewing directions are in simulation coordinates, not
    per-galaxy disc-aligned coordinates. The octant directions (ppp,
    ppm, ..., mmm) align with the body diagonals of the simulation
    coordinate cube.

    Parameters
    ----------
    gal_id: int
        FIREBox galaxy unique ID.
    res: int
        Number of pixels along each axis of the velocity map.
    min_cden: float, default 14.
        Minimum column density in M_sun / pc^2 for a pixel to receive
        a numerical value; pixels below this threshold are np.nan.
    queue: multiprocessing.Queue or None, default None
        When provided, the function sends progress messages of the form
        ('load', gal_id, None), ('proj', gal_id, n), ('done', gal_id,
        None), ('skip', gal_id, reason), and ('error', gal_id, msg).
        save_all_firebox_vmaps uses this for its progress display.

    Returns
    -------
    d: dict or None
        Returns None when the particle file is missing or no bound gas
        particles exist. Otherwise, keys are projection names (e.g.
        'projection_xy', 'projection_ppp'). Each value is a dict with
        the following keys:

        'vmap': np.ndarray, shape (res, res)
            Mass-weighted mean line-of-sight velocity in km/s. Pixels
            whose column density falls below min_cden are np.nan.
        'horiz_edges': np.ndarray, shape (res+1,)
            Bin edges along the horizontal axis in physical kpc.
        'vert_edges': np.ndarray, shape (res+1,)
            Bin edges along the vertical axis in physical kpc.
    '''
    from . import config
    from . import firebox_io
    from . import rotate_galaxy
    import os
    import h5py
    import numpy as np

    firebox_dir = config.config[f'{__package__}_paths']['firebox_data_dir']
    firebox_snap = config.config[f'{__package__}_paths']['firebox_snap']
    output_dir = os.path.join(
        config.config[f'{__package__}_paths']['project_data_dir'],
        'vmaps_res{0:0.0f}_min_cden{1:0.1e}'.format(res, min_cden),
    )
    obj_path = os.path.join(
        firebox_dir,
        firebox_snap,
        f'particles_within_Rvir_object_{gal_id}.hdf5',
    )
    output_path = os.path.join(
        output_dir,
        f'object_{gal_id}_vmap.hdf5',
    )

    if not os.path.exists(obj_path):
        if queue is not None:
            queue.put(('skip', gal_id, 'file not found'))
        return None

    proj_i = 0
    d = {}
    try:
        coords, vs, ms, _ = firebox_io.load_particles('gas', obj_path)
        if len(coords) == 0:
            if queue is not None:
                queue.put(('skip', gal_id, 'no bound gas'))
            return None

        if queue is not None:
            queue.put(('load', gal_id, None))

        try:
            fov = firebox_io.get_fov(gal_id)
        except KeyError:
            if queue is not None:
                queue.put(('skip', gal_id, 'no FOV data'))
            return None

        # Standard projections: (horiz_axis, vert_axis) index pairs.
        standard = {
            'projection_xy': (0, 1),
            'projection_yz': (1, 2),
            'projection_zx': (2, 0),
        }
        z_hat = np.array([0., 0., 1.])

        os.makedirs(output_dir, exist_ok=True)
        with h5py.File(output_path, 'w') as out_f:
            out_f.attrs['fov']      = fov
            out_f.attrs['min_cden'] = min_cden
            out_f.attrs['res']      = res

            for proj_name, (h_ax, v_ax) in standard.items():
                in_fov = (
                    np.linalg.norm(coords[:, [h_ax, v_ax]], axis=1)
                    <= fov / 2.
                )
                c, v, m = coords[in_fov], vs[in_fov], ms[in_fov]
                if len(c) == 0:
                    continue
                vmap, horiz_edges, vert_edges = calc_vmap(
                    c, v, m, h_ax, v_ax, res, min_cden,
                )
                d[proj_name] = {
                    'vmap': vmap,
                    'horiz_edges': horiz_edges,
                    'vert_edges': vert_edges,
                }
                grp = out_f.create_group(proj_name)
                grp.create_dataset('vmap', data=vmap)
                grp.create_dataset('horiz_edges', data=horiz_edges)
                grp.create_dataset('vert_edges', data=vert_edges)
                proj_i += 1
                if queue is not None:
                    queue.put(('proj', gal_id, proj_i))

            for label, n in rotate_galaxy.OCTANT_DIRECTIONS.items():
                proj_name = f'projection_{label}'
                R = rotate_galaxy.rotation_matrix(n, z_hat)
                coords_rot = coords @ R.T
                vs_rot     = vs    @ R.T
                in_fov = (
                    np.linalg.norm(coords_rot[:, :2], axis=1)
                    <= fov / 2.
                )
                c, v, m = (
                    coords_rot[in_fov], vs_rot[in_fov], ms[in_fov],
                )
                if len(c) == 0:
                    continue
                vmap, horiz_edges, vert_edges = calc_vmap(
                    c, v, m, 0, 1, res, min_cden,
                )
                d[proj_name] = {
                    'vmap': vmap,
                    'horiz_edges': horiz_edges,
                    'vert_edges': vert_edges,
                }
                grp = out_f.create_group(proj_name)
                grp.create_dataset('vmap', data=vmap)
                grp.create_dataset('horiz_edges', data=horiz_edges)
                grp.create_dataset('vert_edges', data=vert_edges)
                proj_i += 1
                if queue is not None:
                    queue.put(('proj', gal_id, proj_i))

    except Exception as exc:
        if os.path.exists(output_path):
            os.remove(output_path)
        if queue is not None:
            queue.put(('error', gal_id, str(exc)))
        return None

    if queue is not None:
        queue.put(('done', gal_id, None))
    return d


def _vmap_worker(args):
    '''Unpack arguments for firebox_vmap for use with Pool.starmap.'''
    return firebox_vmap(*args)


def save_all_firebox_vmaps(res, min_cden=14.):
    '''
    Save bound-gas velocity maps for all FIREBox galaxies whose
    particle files exist in firebox_data_dir/firebox_snap (both set in
    the uci_tools_paths section of the config).
    Each output file contains 11 projection groups: projection_xy,
    projection_yz, projection_zx, and one group per octant direction.
    The code saves files to
    project_data_dir/vmaps_res{res}_min_cden{min_cden}.

    Parameters
    ----------
    res: int
        Number of pixels along each axis of the velocity map.
    min_cden: float, default 14.
        Minimum column density in M_sun / pc^2 for a pixel to receive
        a numerical value; pixels below this threshold are np.nan.

    Returns
    -------
    None
    '''
    from . import firebox_io
    import multiprocessing
    import rich.live
    import rich.progress

    df = firebox_io.load_grp_ids()
    gal_ids = list(df.index)
    n_galaxies = len(gal_ids)

    n_workers = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    work_args = [
        (gal_id, res, min_cden, queue)
        for gal_id in gal_ids
    ]

    progress = rich.progress.Progress(
        rich.progress.TextColumn(
            '{task.description}', style='bold',
        ),
        rich.progress.BarColumn(bar_width=30),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeElapsedColumn(),
    )

    overall_task = progress.add_task(
        f'[cyan]Overall ({n_workers} workers)',
        total=n_galaxies,
    )

    # Maps gal_id -> rich task_id for active galaxy bars.
    active_tasks = {}

    n_done = 0
    n_finished = 0
    # Maps skip/error reason -> list of gal_ids.
    skipped = {}
    errors  = {}

    def _handle_msg(kind, gal_id, payload):
        nonlocal n_done, n_finished
        if kind == 'load':
            tid = progress.add_task(
                f'  galaxy {gal_id}',
                total=11,
            )
            active_tasks[gal_id] = tid

        elif kind == 'proj':
            tid = active_tasks.get(gal_id)
            if tid is not None:
                progress.update(
                    tid, completed=payload,
                )

        elif kind == 'done':
            tid = active_tasks.pop(gal_id, None)
            if tid is not None:
                progress.update(
                    tid,
                    completed=11,
                    description=(
                        f'  galaxy {gal_id} [green]✓'
                    ),
                )
                progress.remove_task(tid)
            n_done += 1
            n_finished += 1
            progress.update(
                overall_task, completed=n_finished,
            )

        elif kind == 'skip':
            skipped.setdefault(payload, []).append(gal_id)
            n_finished += 1
            progress.update(
                overall_task, completed=n_finished,
            )

        elif kind == 'error':
            tid = active_tasks.pop(gal_id, None)
            if tid is not None:
                progress.update(
                    tid,
                    description=(
                        f'  galaxy {gal_id}'
                        f' [red]✗ {payload}'
                    ),
                )
                progress.remove_task(tid)
            errors.setdefault(payload, []).append(gal_id)
            n_finished += 1
            progress.update(
                overall_task, completed=n_finished,
            )

    with (
        multiprocessing.Pool(n_workers) as pool,
        rich.live.Live(progress, refresh_per_second=12),
    ):
        # pool.starmap_async is from Python's multiprocessing package.
        # Like the * operator, it unpacks each tuple in work_args and
        # calls firebox_vmap(*args) for each one across the worker pool.
        # The _async suffix means it returns immediately rather than
        # blocking, so the while loop below can drain the progress queue
        # while workers run.
        async_results = pool.starmap_async(
            firebox_vmap, work_args,
        )

        while not async_results.ready() or not queue.empty():
            try:
                msg = queue.get(timeout=0.1)
            except Exception:
                continue
            _handle_msg(*msg)

        # Drain any remaining messages after the pool finishes.
        while not queue.empty():
            try:
                _handle_msg(*queue.get_nowait())
            except Exception:
                break

    n_skip = sum(len(v) for v in skipped.values())
    n_err  = sum(len(v) for v in errors.values())
    print(f'\nDone. {n_done} processed, {n_skip} skipped,'
          f' {n_err} errors.')
    if skipped:
        print('\nSkipped:')
        for reason, ids in skipped.items():
            print(f'  {reason} ({len(ids)}): {ids}')
    if errors:
        print('\nErrors:')
        for reason, ids in errors.items():
            print(f'  {reason} ({len(ids)}): {ids}')
    return None


def load_firebox_vmap(gal_id, res, min_cden):
    '''
    Display the bound-gas velocity map that `save_all_firebox_vmaps` generated
    for the given galaxy.

    Parameters
    ----------
    gal_id: int
        FIREBox galaxy unique ID
    res: int
        The number of pixels the velocity map has along each axis.
    min_cden: float, default 14.
        The minimum column density in M_sun / pc^2 with which
        `save_all_firebox_vmaps` generated the velocity map. That function sets
        pixels that are below the minimum density to np.nan.

    Returns
    -------
    None
    '''
    from . import config
    import os
    import h5py
    import numpy as np
    from matplotlib import pyplot as plt
    maps_dir = os.path.join(
        config.config[f'{__package__}_paths']['project_data_dir'],
        'vmaps_res{0:0.0f}_min_cden{1:0.1e}'.format(res, min_cden)
    )

    orientation_d = {
        'projection_xy': {'h': 0, 'v': 1},
        'projection_yz': {'h': 1, 'v': 2},
        'projection_zx': {'h': 2, 'v': 0}
    }
    axes_d = {0: '$x$', 1: '$y$', 2: '$z$'}

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True)

    path = os.path.join(maps_dir, f'object_{gal_id}_vmap.hdf5')
    with h5py.File(path, 'r') as f:
        for i, (name, obj) in enumerate(f.items()):
            if isinstance(obj, h5py.Group):
                vmap = obj['vmap'][()]
                horiz_edges = obj['horiz_edges'][()]
                vert_edges = obj['vert_edges'][()]
                
                vmax = np.nanmax(vmap)
                vmin = -1. * vmax

                quadmesh = axs[i].pcolormesh(
                    horiz_edges,
                    vert_edges,
                    vmap,
                    cmap=plt.cm.seismic_r,
                    vmin=vmin,
                    vmax=vmax
                )
                axs[i].set_xlabel(
                    '{0} [kpc]'.format(axes_d[orientation_d[name]['h']])
                )
                axs[i].set_ylabel(
                    '{0} [kpc]'.format(axes_d[orientation_d[name]['v']])
                )
                axs[i].set_aspect('equal', adjustable='box')
                fig.colorbar(
                    quadmesh,
                    ax=axs[i],
                    label=r'Gas LOS Velocity [kms$^{-1}]$'
                )
    return None


def imshow_firebox_vmap(gal_id, res, min_cden):
    '''
    Display the bound-gas velocity map that `save_all_firebox_vmaps` generated
    for the given galaxy.

    Parameters
    ----------
    gal_id: int
        FIREBox galaxy unique ID
    res: int
        The number of pixels the velocity map has along each axis.
    min_cden: float, default 14.
        The minimum column density in M_sun / pc^2 with which
        `save_all_firebox_vmaps` generated the velocity map. That function sets
        pixels that are below the minimum density to np.nan.

    Returns
    -------
    None
    '''
    from . import config
    import os
    import h5py
    import numpy as np
    from matplotlib import pyplot as plt
    maps_dir = os.path.join(
        config.config[f'{__package__}_paths']['project_data_dir'],
        'vmaps_res{0:0.0f}_min_cden{1:0.1e}'.format(res, min_cden)
    )

    orientation_d = {
        'projection_xy': {'h': 0, 'v': 1},
        'projection_yz': {'h': 1, 'v': 2},
        'projection_zx': {'h': 2, 'v': 0}
    }
    axes_d = {0: '$x$', 1: '$y$', 2: '$z$'}

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True)

    path = os.path.join(maps_dir, f'object_{gal_id}_vmap.hdf5')
    with h5py.File(path, 'r') as f:
        for i, (name, obj) in enumerate(f.items()):
            if isinstance(obj, h5py.Group):
                vmap = obj['vmap'][()]
                horiz_edges = obj['horiz_edges'][()]
                vert_edges = obj['vert_edges'][()]
                
                vmax = np.nanmax(vmap)
                vmin = -1. * vmax

                axs[i].imshow(vmap)
                axs[i].set_xlabel(
                    '{0} [kpc]'.format(axes_d[orientation_d[name]['h']])
                )
                axs[i].set_ylabel(
                    '{0} [kpc]'.format(axes_d[orientation_d[name]['v']])
                )
                axs[i].set_aspect('equal', adjustable='box')
    plt.show()
    return None


def show_firebox_vmap_live(gal_id, res, min_cden=14.):
    '''
    Compute and display the bound-gas velocity map for a FIREBox galaxy
    without reading from or writing to a file.

    Parameters
    ----------
    gal_id: int
        FIREBox galaxy unique ID.
    res: int
        Number of pixels along each axis of the velocity map.
    min_cden: float, default 14.
        Minimum column density in M_sun / pc^2 for a pixel to receive a
        numerical value; pixels below this threshold are np.nan.

    Returns
    -------
    None
    '''
    import numpy as np
    from matplotlib import pyplot as plt

    d = firebox_vmap(gal_id, res, min_cden)
    if d is None:
        print(f'No data for galaxy {gal_id}.')
        return None

    orientation_d = {
        'projection_xy': {'h': 0, 'v': 1},
        'projection_yz': {'h': 1, 'v': 2},
        'projection_zx': {'h': 2, 'v': 0},
    }
    axes_d = {0: '$x$', 1: '$y$', 2: '$z$'}

    projections = list(d.keys())
    n = len(projections)
    fig, axs = plt.subplots(4, 3, figsize=(12, 16))
    axs = axs.ravel()
    for ax in axs[n:]:
        ax.set_visible(False)

    for i, name in enumerate(projections):
        vmap = d[name]['vmap']
        horiz_edges = d[name]['horiz_edges']
        vert_edges = d[name]['vert_edges']
        vmax = np.nanmax(np.abs(vmap))
        quadmesh = axs[i].pcolormesh(
            horiz_edges,
            vert_edges,
            vmap,
            cmap=plt.cm.seismic_r,
            vmin=-vmax,
            vmax=vmax,
        )
        if name in orientation_d:
            axs[i].set_xlabel(
                '{0} [kpc]'.format(axes_d[orientation_d[name]['h']])
            )
            axs[i].set_ylabel(
                '{0} [kpc]'.format(axes_d[orientation_d[name]['v']])
            )
        else:
            axs[i].set_xlabel('kpc')
            axs[i].set_ylabel('kpc')
        label = name.replace('projection_', '')
        axs[i].set_title(label)
        axs[i].set_aspect('equal', adjustable='box')
        fig.colorbar(
            quadmesh,
            ax=axs[i],
            label=r'Gas LOS Velocity [km s$^{-1}$]',
        )
    plt.tight_layout()
    plt.show()
    return None
