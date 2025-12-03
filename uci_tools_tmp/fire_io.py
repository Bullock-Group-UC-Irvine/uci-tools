import os
import warnings
from . import rotate_galaxy
from . import tools as uci
from . import staudt_tools
from progressbar import ProgressBar

class mydict(dict):
    # Need this so I can add attributes to a dictionary
    pass

def gen_gal_data(
        galname,
        min_radius=None,
        max_radius=None,
        save=False,
        cropped_run=None):
    '''
    Generate cropped data from the original hdf5 files for a particular
    galaxy. Data is cropped at a
    certain radius from the center of the galaxy. The data is saved in 
    h5py.Group's corresponding to the particle type. 

    Particle types are
        'PartType0' is gas. 
        'PartType1' is dark matter.
        'PartType2' is dummy collisionless. 
        'PartType3' is grains/PIC particles. 
        'PartType4' is stars. 
        'PartType5' is black holes / sinks.

    In each particle-type h5py.Group, there are h5py.Dataset's. I provide an
    incomplete description of the Datasets here:
        'mass_phys': Physical mass of each particle in units of 10^10 M_sun
        'v_vec_rot': Velocity vector in Cartesian coordinates where the
            velocities have all been uniformly rotated so the z component
            points in the direction of the galaxy's angular momentum vector.
        'v_vec_disc': Only x and y components of velocity
        'coord_disc': Only x and y components of coordinates

        'v_dot_rhat': The r (scalar) component of velocity, where 
            d['coord_disc'] gives
            the r vector (can be negative if the projection of velocity along r
            points in the opposite direction of r)
        'v_dot_phihat': The phi (scalar) component of velocity (can be negative
            if the projection of velocity along phi points in the opposite
            direction of phi)
        'v_dot_zhat': The z (scalar) component of velocity (can be negative if
            the projection of velocity along z points in the opposite direction
            of z)

        'v_r_vec': The projection of velocity along the r vector, expressed in
            Cartesian x and y components
        'v_phi_vec': The projection of velocity along the phi vector, expressed
            in Cartesian x and y components
        'v_phi_mag': The magnitude of the projection of velocity along the phi
            vector (always positive)

    Parameters
    ----------
    galname: str
        Name of the galaxy for which the method should generate cropped data
    min_radius: float, default None
        Particles below this radius in kpc will be excluded. The user will use
        this for generating cropped data.
    max_radius: float, default None
        Particles beyond this radius in kpc will be exluded. The user will use
        this for generating cropped data.
    save: bool, default False
        Whether to save the run into /DFS-L/DATA/cosmo/pstaudt/FIRE. This is
        only relevent when we're cropping the data. If save == True, the user
        must also specify a cropped_run name.
    cropped_run: str
        The suffix of the directory into which the method should save the data.
        For example, the directory containing cropped data that was rotated 
        based on all 3 of stars, DM, and gas is 'GVB_202304'. In this case,
        `cropped_run` was '202304'. The directory containing cropped data
        rotated based on only stars is 'GVB_star_rot'. In this case, 
        `cropped_run` was 'star_rot'.
    '''
    import h5py
    import numpy as np

    if save and cropped_run is None:
        raise ValueError(
            'If `save` is True, the user must specify a `cropped_run` name.'
        )
    if (
            (min_radius is not None or max_radius is not None) 
            and not (min_radius is not None and max_radius is not None)):
        raise ValueError(
                'You must specify either both `min_radius` and `max_radius` or'
                ' neither.'
            )

    print('Generating {0:s} data'.format(galname))

    df = staudt_tools.init_df()

    suffix=df.loc[galname,'fsuffix']
    suffix_cropped = df.loc[galname, 'fsuffix_cropped']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    typ = 'fire'

    #original directiory result:
    orig_dir_res = staudt_tools.build_direcs(
            suffix, res, mass_class, typ,
            source='original', 
            cropped_run=cropped_run
    )
    halodirec, snapdir_orig, almost_full_path_orig, num_files = orig_dir_res

    d = mydict() 
    pbar=ProgressBar()
    for i in pbar(range(0, num_files)):
        with h5py.File(almost_full_path_orig+'.'+str(i)+'.hdf5', 'r') as f:
            for key1 in f.keys():
                if key1 == 'Header': 
                    if hasattr(d, 'attrs'):
                        # We only need to save the header attributes once. 
                        # They'll 
                        # be the same
                        # in all snapshot files.
                        continue # Move onto the next key1.
                    else:
                        d.attrs = {}
                        for attr in f[key1].attrs.keys():
                            # Extract the data from the 
                            # `h5py._hl.attrs.AttributeManager`
                            d.attrs[attr] = f[key1].attrs[attr]
                        continue #move onto the next key1
                if key1 not in d:
                    #If this is the first run, initialize the 
                    #sub-dictionary
                    #corresponding to key1:
                    d[key1]=dict()
                for key2 in f[key1].keys():
                    #Extract the data from the `h5py._hl.group.Group`s
                    new_data = f[key1][key2][:]
                    if key2 in d[key1]:
                        #If this isn't the first snapshot subfile, add the 
                        #new
                        #data to the data as of the last subfile.
                        d[key1][key2] = np.concatenate((d[key1][key2], 
                                                        new_data))
                    else:
                        d[key1][key2] = new_data

    # Getting host halo info
    center_coord, rvir, v_halo, mvir = uci.get_halo_info(
        halodirec,
        suffix, 
        typ, 
        host_key, 
        mass_class
    )

    h = d.attrs['HubbleParam']
    for key1 in d.keys():
        d[key1]['coord_phys'] = d[key1]['Coordinates']/h
        d[key1]['coord_centered'] = d[key1]['coord_phys']-center_coord
        d[key1]['v_vec_centered'] = d[key1]['Velocities']-v_halo
        d[key1]['r'] = np.linalg.norm(d[key1]['coord_centered'], axis=1)
        d[key1]['mass_phys'] = d[key1]['Masses']/h #units of 1e10 M_sun

        if min_radius is not None or max_radius is not None:
            within_crop = (d[key1]['r'] <= max_radius) \
                           & (d[key1]['r'] >= min_radius)
            for key2 in d[key1].keys():
                d[key1][key2] = d[key1][key2][within_crop] #crop the dictionary
    
    #calculate temperatures
    he_fracs = d['PartType0']['Metallicity'][:,1]
    d['PartType0']['T'] = uci.calc_temps(he_fracs,
                                            *[d['PartType0'][key] \
                                              for key in \
                                              ['ElectronAbundance',
                                               'InternalEnergy']])

    # Get the rotation matrix using only stars.
    rotation_matrix = rotate_galaxy.rotation_matrix_fr_dat(
        *[flatten_particle_data(
              d,
              data,
              drop_particles=['PartType0',
                              'PartType1',
                              'PartType2',
                              'PartType3',
                              'PartType5'],
              supress_warning=True)
          for data in ['coord_centered',
                       'v_vec_centered',
                       'mass_phys',
                       'r']]
    )

    for ptcl in d.keys(): #for each particle type
        print('Rotating {0:s}'.format(ptcl))
        d[ptcl]['v_vec_rot'] = rotate_galaxy.rotate(
                d[ptcl]['v_vec_centered'], rotation_matrix)
        d[ptcl]['coord_rot'] = rotate_galaxy.rotate(
                d[ptcl]['coord_centered'], rotation_matrix)
        # Add v_dot_phihat and other cylindrical velocity information
        d[ptcl] = d[ptcl] | uci.calc_cyl_vels(d[ptcl]['v_vec_rot'],
                                              d[ptcl]['coord_rot'])
        d[ptcl]['v_dot_zhat'] = d[ptcl]['v_vec_rot'][:,2]

        l = len(d[ptcl]['mass_phys'])
        for key in d[ptcl].keys():
            try:
                assert len(d[ptcl][key])==l
            except:
                print('failed on '+key)

    if save:
        #cropped directory result:
        crop_dir_res = staudt_tools.build_direcs(
            suffix_cropped, 
            res,
            mass_class,
            typ,
            source='cropped',
            min_radius=min_radius,
            max_radius=max_radius,
            cropped_run=cropped_run
        )
        _, snapdir_crop, almost_full_path_crop, _ = crop_dir_res
        if not os.path.isdir(snapdir_crop):
            os.makedirs(snapdir_crop)
        with h5py.File(almost_full_path_crop+'.hdf5',
                       'w') as f_crop:
            header = f_crop.create_group('Header')
            for attr in d.attrs:
                header.attrs[attr] = d.attrs[attr]
            for ptcl in d.keys(): #for each particle type
                f_crop.create_group(ptcl)
                for key2 in d[ptcl].keys():
                    print('saving {0:s}, {1:s}'.format(ptcl,key2))
                    data = d[ptcl][key2]
                    f_crop[ptcl].create_dataset(
                        key2,
                        data=data,
                        dtype=data.dtype
                    )
            print('')

    return d 

def gen_all_gals_data(run_name):
    '''
    Generate cropped data for all m12 galaxies from the original hdf5 files.
    Data is cropped at 10 kpc from the center of the galaxies.

    Parameters
    ----------
    run_name: str
        The suffix of the directory into which the method should save the data.
        For example, the directory containing cropped data that was rotated 
        based on all 3 of stars, DM, and gas is 'GVB_202304'. In this case,
        `cropped_run` was '202304'. The directory containing cropped data
        rotated based on only stars is 'GVB_star_rot'. In this case, 
        `cropped_run` was 'star_rot'.
    '''
    import h5py
    import numpy as np

    df = staudt_tools.init_df()
    for gal in df.index:
        gen_gal_data(
            gal,
            min_radius=0.,
            max_radius=10.,
            save=True,
            cropped_run=run_name
        )
    
    return None

def flatten_particle_data(
        d,
        data,
        drop_particles=['PartType2'],
        supress_warning=False):
    '''
    Given a galaxy dictionary layered by particle in the same way as the 
    original hdf5
    files, return data from that dictionary for all particle types,
    marginalizing particle type information.

    Parameters:
        d: dict 
            Galaxy dictionary
        data: str
            The key corresponding to the data the user wants to
            extract
        drop_particles: list of str
            The particle types that should not be included in the flattened
            data

            'PartType0' is gas. 'PartType1' is dark matter.
            'PartType2' is dummy collisionless. 'PartType3' is grains/PIC
            particles. 'PartType4' is stars. 'PartType5' is black holes / 
            sinks.
    '''
    import h5py
    import numpy as np

    keys = list(d.keys())
    try:
        keys.remove('Header')
    except:
        pass
    for part in drop_particles:
        if part not in keys:
            if not supress_warning:
                warnings.warn(
                    '\n{0:s} is not in the data, but the user'
                    ' attempted to exclude it from a calculation. Ensure'
                    ' there is not a typo in the `drop_particles` list.'
                    ' Otherwise the code may be including a particle'
                    ' type that should be excluded.'.format(part)
                )
        else:
            keys.remove(part)
    data_flat = np.concatenate([d[parttype][data] for parttype in keys])

    return data_flat

def load_cropped_data(galname, getparts='all', verbose=True, cropped_run=None):
    '''
    Load data from cropped hdf5 files, as opposed to theoriginal hdf5 files.

    Parameters
    ----------
    galname: str
        The galaxy name string corresponding to an index in df.
    getparts: str or list of str, default 'all' 
        Specifies the particle types to extract
        'PartType0' is gas. 
        'PartType1' is dark matter.
        'PartType2' is dummy collisionless. 
        'PartType3' is grains/PIC particles. 
        'PartType4' is stars. 
        'PartType5' is black holes / sinks.
    verbose: bool, default True
        If True, the function prints which galaxy its pulling with a progress
        bar.
    cropped_run: str
        Specify the name of the run that generated the cropped files of
        interest. This is a kwarg, but it is mandatory that the user provide
        it.
        For example, the directory containing cropped data that was rotated 
        based on all 3 of stars, DM, and gas is 'GVB_202304'. In this case,
        `cropped_run` was '202304'. The directory containing cropped data
        rotated based on only stars is 'GVB_star_rot'. In this case, 
        `cropped_run` was 'star_rot'.

    Returns
    -------
    d: dict
        Galaxy dictionary split by particle type

        Particle types are
            'PartType0' is gas. 
            'PartType1' is dark matter.
            'PartType2' is dummy collisionless. 
            'PartType3' is grains/PIC particles. 
            'PartType4' is stars. 
            'PartType5' is black holes / sinks.

        I provide an
        incomplete description of the dictionaries' key, value pairs here:
            'mass_phys': Physical mass of each particle in units of 10^10 M_sun

            'v_vec_rot': Velocity vector in Cartesian coordinates where the
                velocities have all been uniformly rotated so the z component
                points in the direction of the galaxy's angular momentum
                vector.
            'v_vec_disc': Only x and y components of velocity
            'coord_disc': Only x and y components of coordinates

            'v_dot_rhat': The r (scalar) component of velocity, where 
                d['coord_disc'] gives
                the r vector (can be negative if the projection of velocity
                along r
                points in the opposite direction of r)
            'v_dot_phihat': The phi (scalar) component of velocity (can be
                negative
                if the projection of velocity along phi points in the opposite
                direction of phi)
            'v_dot_zhat': The z (scalar) component of velocity (can be negative
                if
                the projection of velocity along z points in the opposite 
                direction
                of z)

            'v_r_vec': The projection of velocity along the r vector, expressed
                in
                Cartesian x and y components
            'v_phi_vec': The projection of velocity along the phi vector,
                expressed
                in Cartesian x and y components
            'v_phi_mag': The magnitude of the projection of velocity along the 
                phi
                vector (always positive)
    '''
    import h5py
    import numpy as np

    if verbose:
        print('Loading {0:s}'.format(galname))

    from uci_tools.staudt_tools import build_direcs 

    min_radius = 0. #kpc
    max_radius = 10. #kpc

    df = staudt_tools.init_df()

    suffix=df.loc[galname,'fsuffix']
    suffix_cropped = df.loc[galname, 'fsuffix_cropped']
    res=df.loc[galname,'res']
    host_key=df.loc[galname,'host_key']
    mass_class = df.loc[galname,'mass_class']
    typ = 'fire'

    #cropped directory result
    crop_dir_res = build_direcs(
        suffix_cropped,
        res,
        mass_class,
        typ,
        source='cropped',
        min_radius=0.,
        max_radius=10.,
        cropped_run=cropped_run)
    _, snapdir_crop, almost_full_path_crop, _ = crop_dir_res

    with h5py.File(almost_full_path_crop+'.hdf5',
                   'r') as f:
        d = mydict()
        d.attrs = {}
        for attr in f['Header'].attrs.keys():
            d.attrs[attr] = f['Header'].attrs[attr]
        
        if getparts=='all':
            # All the particle types will be the keys in f, except for the
            # 'Header' key.
            key1s = list(f.keys())
            key1s.remove('Header')
        else: 
            key1s = getparts
        if verbose:
            pbar = ProgressBar()
        else:
            #pbar wrapper does nothing
            pbar = lambda function: function
        for ptcl in pbar(key1s):
            d[ptcl] = {}
            key2s = f[ptcl].keys()
            for datapnt in key2s:
                d[ptcl][datapnt] = f[ptcl][datapnt][:]
    return d
