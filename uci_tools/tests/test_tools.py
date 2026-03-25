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

class TestRotateGalaxy(unittest.TestCase):
    '''
    Test uci_tools.rotate_galaxy.rotation_matrix_fr_vecs and related
    helpers.  The non-degenerate test cases compare the output against
    reference values that save_rotation_matrix_data generated from the
    old vrrotvec/vrrotvec2mat chain before the rewrite.
    '''

    REF_PATH = os.path.join(
        os.path.dirname(__file__),
        'test_data',
        'rotation_matrices.hdf5',
    )

    def _load_ref(self, name):
        import h5py
        with h5py.File(self.REF_PATH, 'r') as f:
            a = f[name]['a'][()]
            b = f[name]['b'][()]
            R_expected = f[name]['R'][()]
        return a, b, R_expected

    def test_x_to_z_matches_old(self):
        a, b, R_expected = self._load_ref('x_to_z')
        R = uci.rotate_galaxy.rotation_matrix_fr_vecs(a, b)
        npt.assert_allclose(R, R_expected, atol=1e-12)

    def test_y_to_x_matches_old(self):
        a, b, R_expected = self._load_ref('y_to_x')
        R = uci.rotate_galaxy.rotation_matrix_fr_vecs(a, b)
        npt.assert_allclose(R, R_expected, atol=1e-12)

    def test_ppp_to_z_matches_old(self):
        a, b, R_expected = self._load_ref('ppp_to_z')
        R = uci.rotate_galaxy.rotation_matrix_fr_vecs(a, b)
        npt.assert_allclose(R, R_expected, atol=1e-12)

    def test_unnorm_to_y_matches_old(self):
        # Verifies that unnormalized inputs give the same result
        # as the old code, which normalized internally.
        a, b, R_expected = self._load_ref('unnorm_to_y')
        a_unnorm = np.array([3., 0., 4.])
        R = uci.rotate_galaxy.rotation_matrix_fr_vecs(a_unnorm, b)
        npt.assert_allclose(R, R_expected, atol=1e-12)

    def test_parallel_returns_identity(self):
        # Old code produced NaN for parallel inputs. New code returns
        # identity.
        a = np.array([0., 0., 1.])
        R = uci.rotate_galaxy.rotation_matrix_fr_vecs(a, a)
        npt.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_antiparallel_maps_a_to_b(self):
        # Old code produced NaN for antiparallel inputs. Verify that
        # new code returns a valid rotation.
        a = np.array([0., 0., -1.])
        b = np.array([0., 0.,  1.])
        R = uci.rotate_galaxy.rotation_matrix_fr_vecs(a, b)
        npt.assert_allclose(R @ a, b, atol=1e-12)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)

    def test_result_maps_a_to_b(self):
        # Property test: R @ a_hat should equal b_hat for all cases.
        test_cases = [
            (np.array([1., 0., 0.]), np.array([0., 1., 0.])),
            (np.array([1., 1., 1.]), np.array([0., 0., 1.])),
            (np.array([1., 2., 3.]), np.array([-1., 0., 2.])),
        ]
        for a, b in test_cases:
            a_hat = a / np.linalg.norm(a)
            b_hat = b / np.linalg.norm(b)
            R = uci.rotate_galaxy.rotation_matrix_fr_vecs(a, b)
            npt.assert_allclose(R @ a_hat, b_hat, atol=1e-12)

    def test_result_is_orthogonal(self):
        a = np.array([1., 2., 3.])
        b = np.array([4., 5., 6.])
        R = uci.rotate_galaxy.rotation_matrix_fr_vecs(a, b)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_result_is_proper_rotation(self):
        a = np.array([1., 2., 3.])
        b = np.array([4., 5., 6.])
        R = uci.rotate_galaxy.rotation_matrix_fr_vecs(a, b)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)

    def test_rotation_matrix_alias(self):
        # rotation_matrix is an alias for rotation_matrix_fr_vecs.
        a = np.array([1., 0., 0.])
        b = np.array([0., 0., 1.])
        R1 = uci.rotate_galaxy.rotation_matrix_fr_vecs(a, b)
        R2 = uci.rotate_galaxy.rotation_matrix(a, b)
        npt.assert_array_equal(R1, R2)

    def test_rotation_matrix_to_z(self):
        n = np.array([1., 1., 1.]) / np.sqrt(3.)
        R = uci.rotate_galaxy.rotation_matrix_to_z(n)
        z_hat = np.array([0., 0., 1.])
        npt.assert_allclose(R @ n, z_hat, atol=1e-12)

    def test_rotate_snapdict_rotates_coordinates(self):
        coords = np.array([[1., 0., 0.], [0., 1., 0.]])
        vels = np.array([[0., 1., 0.], [0., 0., 1.]])
        sd = {'Coordinates': coords, 'Velocities': vels, 'Masses': 1.0}
        R = uci.rotate_galaxy.rotation_matrix_to_z(
            np.array([1., 0., 0.])
        )
        result = uci.rotate_galaxy.rotate_snapdict(sd, R)
        npt.assert_allclose(
            result['Coordinates'][0], [0., 0., 1.], atol=1e-12,
        )
        npt.assert_allclose(
            result['Velocities'][0], [0., 1., 0.], atol=1e-12,
        )

    def test_rotate_snapdict_preserves_scalars(self):
        sd = {
            'Coordinates': np.array([[1., 0., 0.]]),
            'Masses': np.array([42.]),
            'r': np.array([1.]),
        }
        R = np.eye(3)
        result = uci.rotate_galaxy.rotate_snapdict(sd, R)
        assert result['Masses'] is sd['Masses']
        assert result['r'] is sd['r']

    def test_rotate_snapdict_absent_key_skipped(self):
        # rotate_snapdict must skip vector keys absent from the dict.
        sd = {'Coordinates': np.array([[1., 0., 0.]])}
        R = uci.rotate_galaxy.rotation_matrix_to_z(
            np.array([1., 0., 0.])
        )
        result = uci.rotate_galaxy.rotate_snapdict(sd, R)
        assert 'Velocities' not in result

    def test_octant_directions_unit_vectors(self):
        for label, n in uci.rotate_galaxy.OCTANT_DIRECTIONS.items():
            self.assertAlmostEqual(
                float(np.linalg.norm(n)), 1.0, places=12,
                msg=f'OCTANT_DIRECTIONS[{label!r}] is not a unit vector',
            )

    def test_octant_directions_cover_all_signs(self):
        labels = set(uci.rotate_galaxy.OCTANT_DIRECTIONS.keys())
        assert labels == {
            'ppp', 'ppm', 'pmp', 'pmm', 'mpp', 'mpm', 'mmp', 'mmm',
        }


class TestFireIO(unittest.TestCase):
    '''
    Test some functions in firebox_io
    '''

    def test_fov(self):
        assert uci.firebox_io.get_fov(0) == 28
        return None

@pytest.fixture(scope='session', autouse=True)
def run_tests():
    # The project_data_dir specified in the ci_config.ini is meant to be temporary.
    # We'll
    # delete it at the end of the code unless by coincidence it exists already.
    output_dir = uci.config.config[f'uci_tools_paths']['project_data_dir']
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
