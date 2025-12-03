#from mock import patch
import numpy as np
import numpy.testing as npt
import unittest
import h5py
import os
import pytest
import shutil

import uci_tools.tools as tools
import uci_tools as uci

class TestLoadFIREData( unittest.TestCase ):
    '''Test suite for loading FIRE data.
    '''

    ###########################################################################

    def test_simple( self ):
        '''This test just ensures that the data can be loaded at all.'''
        import os

        sim_dir = (
            './uci_tools/tests/test_data/downsampled_sim_data/fire_sim/output'
        )
        snapshot = 600
        path = os.path.join(
            sim_dir,
            'snapdir_' + str(snapshot),
            'snapshot_' + str(snapshot) + '.0.hdf5'
        )

        particle_type = 'PartType0'

        snapshot = tools.read_snapshot_simple(
            path,
            particle_type = particle_type,
        )

        assert 'Density' in snapshot.columns
        assert 'Coordinates0' in snapshot.columns

class TestMisc( unittest.TestCase  ):
    ''' Testing nonloaded data
    '''
	
    ########################################################################

    def test_sft_to_ages( self ):
        npt.assert_allclose(tools.sft_to_ages(1), 0, atol = .15) #snapshot 600
        npt.assert_allclose(
                tools.sft_to_ages(0.8550955),
                13.79874688-11.69441659,
                atol = .15
            ) #snapshot 500
        npt.assert_allclose(
                tools.sft_to_ages(0.6958599),
                13.79874688-9.19969494,
                atol = .15
            ) #snapshot 400
        npt.assert_allclose(
                tools.sft_to_ages(0.5366242),
                13.79874688-6.58906279,
                atol = .15
            ) #snapshot 300
        npt.assert_allclose(
                tools.sft_to_ages(0.3777778),
                13.79874688-4.04069309,
                atol = .15
            ) #snapshot 200
        npt.assert_allclose(
                tools.sft_to_ages(0.2187500),
                13.79874688-1.81321181,
                atol = .15
            ) #snapshot 100
        npt.assert_allclose(
                tools.sft_to_ages(0.0100000),
                13.79874688-0.01780470,
                atol = .15
            ) #snapshot 0	

class TestVelMap(unittest.TestCase):
    '''
    Test that velocity map plotter works.
    '''

    def test_vel_map(self):
        data = uci.vel_map.load_m12_data_olti(
            './uci_tools/tests/test_data/downsampled_sim_data/fire_sim/'
                'thelma_downsampled_for_vel_map.h5',
            '600',
        )
        vel_map_output = uci.vel_map.plot(
            *data,
            display_name='Thelma downsampled',
            snap='600',
            horiz_axis=0,
            vert_axis=2,
            res=100,
            min_gas_cden=0.,
            min_stars_cden=0.,
            save_plot=True
        )
        gas_map, young_star_map = vel_map_output[:2]
        with h5py.File(
                './uci_tools/tests/test_data/'
                    'thelma_test_vel_maps.h5',
                'r') as f:
            gas_map_answer = f['velmap_gas'][()]
            young_star_map_answer = f['velmap_stars'][()]
        npt.assert_allclose(gas_map, gas_map_answer)
        npt.assert_allclose(young_star_map, young_star_map_answer)
        return None

class TestFireIO(unittest.TestCase):
    '''
    Test some functions in firebox_io
    '''

    def test_fov(self):
        assert uci.firebox_io.get_fov(0) == 28
        return None

@pytest.fixture(scope='session', autouse=True)
def run_tests():
    # The output_dir specified in the ci_config.ini is meant to be temporary.
    # We'll
    # delete it at the end of the code unless by coincidence it exists already.
    output_dir = uci.config.config[f'uci_tools_paths']['output_dir']
    if not os.path.isdir(output_dir):
        temp_dir = True
        os.mkdir(output_dir)
    else:
        temp_dir = False

    # Run the tests
    yield

    if temp_dir:
        shutil.rmtree(output_dir)

    return None
