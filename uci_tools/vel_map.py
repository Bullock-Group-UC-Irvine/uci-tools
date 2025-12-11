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

def firebox_vmap(gal_id, res, min_cden=14.):
    '''
    Create a velocity map for bound gas in a given FIREBox galaxy with the same
    field of view as Courtney's image of that galaxy. If the FIREBox data file
    doesn't exist in firebox_data_dir/objects_1200 or if the galaxy has no
    bound gas particles, the function returns None. Note that while
    UCITools.firebox_io.get_avg_sfrs (incidentally in the Julia portion of this
    repo), may
    successfully find bound star particles for a given galaxy, firebox_vmap
    may fail to find
    bound gas particles, or vice versa.

    Parameters
    ----------
    gal_id: int
        FIREBox galaxy unique ID
    res: int
        The number of pixels along each axis the velocity map should have
    min_cden: float, default 14.
        The minimum column density in M_sun / pc^2 of particles for a pixel
        in the velocity map to be given a numerical value. Otherwise the pixel
        is np.nan.

    Returns
    -------
    d: Dict or None
        Velocity map dictionary. If the FIREBox data file
        doesn't exist in firebox_data_dir/objects_1200 or if the galaxy has no
        bound particles, the function returns None.

        Key-value pairs are

        'vmap': np.ndarray, shape (res, res)
            The velocity map data for analysis or for use with 
            `matplotlib.axes.Axis.pcolormesh`. Horizontal data is along the
            1 axis.
            Vertical data is along the 0 axis. The user can directly input this
            into
            `pcolormesh`.
        'horiz_edges': np.ndarray, shape (res,)
            The locations of the edges of the velocity map pixels in
            kpc along
            the
            horizontal axis 
        'vert_edges': np.ndarray, shape (res,)
            The locations of the edges of the velocity map
            pixels in kpc along
            the
            vertical axis.
    z_edges, np.ndarray, shape (res,)
    '''
    from . import config
    from . import firebox_io
    import os
    import h5py
    import numpy as np

    id_str = str(gal_id)
    
    super_dir = config.config[f'{__package__}_paths']['firebox_data_dir']
    output_dir = os.path.join(
        config.config[f'{__package__}_paths']['project_data_dir'],
        'vmaps_res{0:0.0f}_min_cden{1:0.1e}'.format(res, min_cden)
    )

    path = os.path.join(
        super_dir,
        'objects_1200',
        f"particles_within_Rvir_object_{id_str}.hdf5"
    )
    output_path = os.path.join(
        output_dir,
        f'object_{gal_id}_vmap.hdf5'
    )

    orientation_d = {
        'projection_xy': {'h': 0, 'v': 1},
        'projection_yz': {'h': 1, 'v': 2},
        'projection_zx': {'h': 2, 'v': 0}
    }

    d = {}
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            for orientation, axes_d in orientation_d.items():
                # Get data for gas inside the fov of Courtney's images.
                coords, vs, ms, ids, fov = firebox_io.load_particle(
                    'gas',
                    f,
                    gal_id,
                    axes_d['h'],
                    axes_d['v'],
                    only_bound=True
                )
                if len(coords) == 0:
                    # If there are no bound particles, end.
                    return None 
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with h5py.File(output_path, 'a') as out_f:
                    out_f.attrs['fov'] = fov
                    out_f.attrs['min_cden'] = min_cden
                    out_f.attrs['res'] = res

                    axes_d = orientation_d[orientation]
                    vmap, horiz_edges, vert_edges = calc_vmap(
                        coords,
                        vs,
                        ms,
                        axes_d['h'],
                        axes_d['v'],
                        res,
                        min_cden
                    )

                    d[orientation] = {}
                    d[orientation]['vmap'] = vmap
                    d[orientation]['horiz_edges'] = horiz_edges
                    d[orientation]['vert_edges'] = vert_edges

                    grp = out_f.create_group(orientation)
                    grp.create_dataset('vmap', data=vmap)
                    grp.create_dataset('horiz_edges', data=horiz_edges)
                    grp.create_dataset('vert_edges', data=vert_edges)
    else:
        return None
    return d

def save_all_firebox_vmaps(res, min_cden=14.):
    '''
    Save bound-gas velocity maps for all FIREBox galaxies whose files exist in 
    firebox_data_dir/objects_1200. The field of
    view for each map corresponds to the field of view in
    Courtney's mock image for the corresponding galaxy. The code saves the maps
    in
    project_data_dir/vmaps_res{res}_min_cden{min_cden}.

    Parameters
    ----------
    res: int
        The number of pixels along each axis the velocity map should have
    min_cden: float, default 14.
        The minimum column density in M_sun / pc^2 of particles for a pixel
        in the velocity map to be given a numerical value. Otherwise the pixel
        is np.nan.

    Returns
    -------
    None
    '''
    from . import config
    from . import firebox_io
    import os
    import glob
    import tqdm
    df = firebox_io.load_grp_ids()
    for gal_id in tqdm.tqdm(df.index, desc='Generating velocity maps'):
        try:
            firebox_vmap(gal_id, res, min_cden)
        except KeyError:
            print(f'object_{gal_id} is missing')
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
