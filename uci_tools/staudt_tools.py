def build_direcs(suffix, res, mass_class, typ='fire', source='original',
                 min_radius=None, max_radius=None, cropped_run=None):
    '''
    Determine the directory where the halo file lives, the snapshot 600
    directory, and the path to the snapshot files (which are in the snapshot
    directory) excluding the file number suffix and file extension.

    Parameters
    ----------
    suffix: str
        The identifer for the desired galaxy, after the mass class. E.g. 'b'
        for 'm12b' or '_elvis_RomeoJuliet' for 'm12_elvis_RomeoJuliet'.
    res: int
        The resolution of the file the user seeks.
    mass_class: int
        The log10(mass class / M_sun). E.g. 12 for the m12's.
    typ: str, default 'fire'
        The type of simulation. Choices are 'fire' or 'dmo'
    source: str, default 'original'
        The source location of the output files. Choices are 'original' and 
        'cropped'.
        'original' files include all particles at all radii and
        are in 
        /DFS-L/DATA/cosmo/grenache/aalazar/FIRE/. 'cropped' files include only
        particles between `min_radius` and `max_radius` and are located in 
        /DFS-L/DATA/cosmo/grenache/staudt/.
    min_radius: float, default None
        If source is 'cropped', the minimum radius of the files the user is
        interested in. Leave as None if source is 'original'.
    max_radius: float, default None
        If source is 'cropped', the maximum radius of the files the user i
        interested in. Leave as None if source is 'original.
    cropped_run: str
        Specify the name of the run that generated the cropped files of
        interest. The user must provide this if `source == 'cropped'`.
        For example, the directory containing cropped data that was rotated 
        based on all 3 of stars, DM, and gas is 'GVB_202304'. In this case,
        `cropped_run` was '202304'. The directory containing cropped data
        rotated based on only stars is 'GVB_star_rot'. In this case, 
        `cropped_run` was 'star_rot'.

    Returns
    -------
    hdirec: str
        The path to the halo file for the specified galaxy
    snapdir: str
        The directory where the snapshot files live for the specified galaxy
    almost_full_path: str
        The path to the snapshot files (which are in the snapshot directory)
        excluding the file number suffix and file extension.
    num_files: int
        The number of snapshot files in `snapdir`.
    '''
    import os
    import traceback

    if source == 'cropped' and cropped_run is None:
        raise ValueError(
            'The user must specify the cropped_run if the source is'
            ' \'cropped\''
        )
        
    assert typ in ['fire','dmo']
    if typ=='fire':
        typ_char='B'
        if source == 'cropped':
            type_and_run = '_'.join([typ_char, cropped_run])
        else:
            type_and_run = typ_char
    elif typ=='dmo':
        typ_char='D'

    res=str(res)
    mass_class='{0:02d}'.format(mass_class)
    snapnum = str(600)

    if source=='original':
        if min_radius is not None or max_radius is not None:
            raise ValueError('radius limits are not applicable when '
                             'source=\'original\'')
        topdirec = '/DFS-L/DATA/cosmo/grenache/aalazar/'
        cropstr = ''
    elif source=='cropped':
        if min_radius is None or max_radius is None:
            raise ValueError('min and max radii must be specified if '
                             'source=\'cropped\'')
        topdirec = '/DFS-L/DATA/cosmo/grenache/staudt/'
        cropstr = '{0:0.1f}_to_{1:0.1f}_kpc/'.format(min_radius, max_radius)
    else:
        raise ValueError('source should be \'original\' or \'cropped\'')
    if int(mass_class)>10:
        hdirec='/DFS-L/DATA/cosmo/grenache/aalazar/FIRE/GV'+typ_char+\
               '/m'+mass_class+suffix+\
               '_res'+res+\
               '/halo/rockstar_dm/hdf5/halo_600.hdf5'
        direc = (topdirec + 'FIRE/GV' + type_and_run + '/m' + mass_class 
                 + suffix + '_res' + res 
                 + '/output/' + cropstr + 'hdf5/')
    elif int(mass_class)==10:
        raise ValueError('Cannot yet handle log M < 11')
        #The following code will not work, but I've leaving it for future
        #development
        if typ=='dmo':
            raise ValueError('Cannot yet handle DMO for log M < 11')
        #Path to m10x runs
        path = '/DFS-L/DATA/cosmo/rouge/mercadf1/fire2/m10x_runs'
        run = 'h1160816' #input the run name
        haloName = 'm'+mass_class+suffix #input the halo name within this run
        pt = 'PartType1' #You can change this to whatever particle you want
        hdirec=path+run+'/'+haloName+'/halo_pos.txt'
        direc=path+run+'/output/snapshot_'+run+'_Z12_bary_box_152.hdf5'
    else:
        raise ValueError('Cannot yet handle log M < 10')

    snapdir = direc+'snapdir_'+snapnum+'/'
    try:
        num_files=len(os.listdir(snapdir))
    except Exception as e:
        print('Warning: {0} encountered when counting snapshot files.'
              ' Saving `num_files` as None.'.format(type(e).__name__))
        traceback.print_exc()
        num_files=None
    #path to the snapshot directory PLUS the first part of the filename:
    almost_full_path = snapdir+'snapshot_'+snapnum

    return hdirec, snapdir, almost_full_path, num_files

def init_df(mass_class=12):
    '''
    Initialize a dataframe specifying information about the file names and
    resolutions of simulated galaxies of
    a given mass class.

    Parameters
    ----------
    mass_class: {9, 10, 11, 12, 'all'}
        The log10(mass / M_sun) mass class of galaxies in which the user is
        interested. The user can also specify 'all' if they want the dataframe
        to contain information for all masses classes from 9 through 12.

    Returns
    -------
    df: pd.DataFrame
    '''
    import pandas as pd
    import numpy as np

    if not isinstance(mass_class,(int,str)):
        raise ValueError('mass_class should be an integer or \'all\'.')
    suffixes_m9 = ['']
    # It's necessary to have different suffix lists for cropped and not cropped
    # because for cropped, I separate the RJ, RR, RL pair folders.
    suffixes_m9cropped = suffixes_m9.copy() 
    ress_m9 = [250]
    gal_names_m9 = ['m9']
    host_keys_m9 = ['host.index']
    mass_classs_m9 = [9]

    '''
    suffixes_m10 = ['b','c','d','e','f','g','h','i','j','k','l','m','q','v','w',
                'xa','xb','xc','xd','xe','xf','xg','xh','xi',
                'y','z']
    ress_m10 = [500]*12 + [250]*2 + [2100] + [4000]*9 + [2100]*2
    '''
    suffixes_m10 = ['q','v','w','y','z']
    suffixes_m10cropped = suffixes_m10.copy()
    #ress_m10 = [500]*12 + [250]*2 + [2100] + [4000]*9 + [2100]*2
    ress_m10 = [250]*2 + [2100]*3
    gal_names_m10=['m10' + g for g in suffixes_m10]
    host_keys_m10 = ['host.index']*len(suffixes_m10)
    mass_classs_m10 = [10]*len(suffixes_m10)
    assert len(suffixes_m10) == len(ress_m10) == len(host_keys_m10) \
           == len(gal_names_m10) == len(mass_classs_m10)

    #suffixes_m11 = ['a','b','c','d','e','f','g','h','i','q','v']
    #ress_m11 = [2100]*3 + [7100]*2 + [17000]*2 + [7100]*4
    #f and g don't have halo files, so I'm skipping them for now.
    suffixes_m11 = ['a','b','c','d','e','h','i','q','v']
    suffixes_m11cropped = suffixes_m11.copy()
    ress_m11 = [2100]*3 + [7100]*2 + [7100]*4
    gal_names_m11 = ['m11' + g for g in suffixes_m11]
    host_keys_m11 = ['host.index']*len(suffixes_m11)
    mass_classs_m11 = [11]*len(suffixes_m11)
    assert len(suffixes_m11) == len(ress_m11) == len(host_keys_m11) \
           == len(gal_names_m11) == len(mass_classs_m11)

    suffixes_m12 = ['b','c','f','i','m','r','w','z',
                '_elvis_RomeoJuliet',
                '_elvis_RomeoJuliet',
                '_elvis_RomulusRemus',
                '_elvis_RomulusRemus',
                '_elvis_ThelmaLouise',
                '_elvis_ThelmaLouise']
    suffixes_m12cropped = ['b','c','f','i','m','r','w','z',
                '_elvis_Romeo',
                '_elvis_Juliet',
                '_elvis_Romulus',
                '_elvis_Remus',
                '_elvis_Thelma',
                '_elvis_Louise']
    ress_m12 = [7100,7100,7100,7100,7100,7100,7100,4200,
                3500,3500,4000,4000,4000,4000]
    gal_names_m12=['m12' + g for g in suffixes_m12[:8]] + ['Romeo','Juliet',
                                                       'Romulus','Remus',
                                                       'Thelma','Louise']
    host_keys_m12 = ['host.index']*8+['host.index','host2.index']*3
    mass_classs_m12 = [12]*len(suffixes_m12)
    assert len(suffixes_m12) == len(ress_m12) == len(host_keys_m12) \
           == len(gal_names_m12) == len(mass_classs_m12)

    if mass_class in [9,10,11,12]:
        lcls=locals()
        exec('suffixes = suffixes_m'+str(mass_class),globals(),lcls)
        exec('suffixes_cropped = suffixes_m'+str(mass_class)+'cropped',
             globals(),lcls)
        exec('ress = ress_m'+str(mass_class),globals(),lcls)
        exec('gal_names = gal_names_m'+str(mass_class),globals(),lcls)
        exec('host_keys = host_keys_m'+str(mass_class),globals(),lcls)
        exec('mass_classs = mass_classs_m'+str(mass_class),globals(),lcls)
        suffixes=lcls['suffixes']
        suffixes_cropped = lcls['suffixes_cropped']
        ress=lcls['ress']
        gal_names=lcls['gal_names']
        host_keys=lcls['host_keys']
        mass_classs=lcls['mass_classs']
    elif mass_class=='all':
        '''
        suffixes = suffixes_m9 + suffixes_m10 + suffixes_m11 + suffixes_m12
        suffixes_cropped = suffixes_m9cropped + suffixes_m10cropped \
                           + suffixes_m11cropped + suffixes_m12cropped
        ress = ress_m9 + ress_m10 + ress_m11 + ress_m12
        gal_names = gal_names_m9 + gal_names_m10 + gal_names_m11 + gal_names_m12
        host_keys = host_keys_m9 + host_keys_m10 + host_keys_m11 + host_keys_m12
        '''
        #Removing m9 and m10 for now
        suffixes = suffixes_m11 + suffixes_m12
        suffixes_cropped = suffixes_m11cropped + suffixes_m12cropped
        ress = ress_m11 + ress_m12
        gal_names = gal_names_m11 + gal_names_m12
        host_keys = host_keys_m11 + host_keys_m12
        mass_classs = mass_classs_m11 + mass_classs_m12
    else:
        raise ValueError('mass_class must be 9,10,11,12 or \'all\'')
        
    df=pd.DataFrame(index=gal_names,
                    data=np.transpose([suffixes, suffixes_cropped, 
                                       ress, host_keys]),
                    columns=['fsuffix','fsuffix_cropped','res','host_key'])
    #Need to add mass_classs *after* creating df because we need it to have an 
    #int dtype.
    df['mass_class']=mass_classs 

    return df
