# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Parameters of the BOP datasets."""

import math
from os.path import join

from bop_toolkit_lib import inout


def get_camera_params(datasets_path, dataset_name, cam_type=None):
  """Return camera parameters for the specified dataset.

  Note that parameters returned by this functions are meant only for simulation
  of the used sensor when rendering training images. To get per-image camera
  parameters (which may vary), use path template 'scene_camera_tpath' contained
  in the dictionary returned by function get_split_params.

  :param datasets_path: Path to a folder with datasets.
  :param dataset_name: Name of the dataset for which to return the parameters.
  :param cam_type: Type of camera.
  :return: Dictionary with camera parameters for the specified dataset.
  """
  # T-LESS includes images captured by three sensors.
  if dataset_name == 'tless':

    # Use images from the Primesense sensor as default.
    if cam_type is None:
      cam_type = 'primesense'
    cam_filename = 'camera_{}.json'.format(cam_type)

  else:
    cam_filename = 'camera.json'

  # Path to the camera file.
  cam_params_path = join(datasets_path, dataset_name, cam_filename)

  p = {
    # Path to a file with camera parameters.
    'cam_params_path': cam_params_path,
  }

  # Add a dictionary containing the intrinsic camera matrix ('K'), image size
  # ('im_size'), and scale of the depth images ('depth_scale', optional).
  p.update(inout.load_cam_params(cam_params_path))

  return p


def get_model_params(datasets_path, dataset_name, model_type=None):
  """Return parameters of object models for the specified dataset.

  :param datasets_path: Path to a folder with datasets.
  :param dataset_name: Name of the dataset for which to return the parameters.
  :param model_type: Type of object models.
  :return: Dictionary with object model parameters for the specified dataset.
  """
  # Object ID's.
  obj_ids = {
    'lm': range(1, 16),
    'lmo': [1, 5, 6, 8, 9, 10, 11, 12],
    'tless': range(1, 31),
    'tudl': range(1, 4),
    'tyol': range(1, 22),
    'ruapc': range(1, 15),
    'icmi': range(1, 7),
    'icbin': range(1, 3),
    'itodd': range(1, 29),
    'hb': [1, 3, 4, 8, 9, 10, 12, 15, 17, 18, 19, 22, 23, 29, 32, 33],
    # 'hb': range(1, 34),  # Original HB dataset.
  }[dataset_name]

  # ID's of objects with symmetries.
  symmetric_obj_ids = {
    'lm': [3, 7, 10, 11],
    'lmo': [10, 11],
    'tless': range(1, 31),
    'tudl': [],
    'tyol': [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21],
    'ruapc': [8, 9, 12, 13],
    'icmi': [1, 2, 6],
    'icbin': [1],
    'itodd': [2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 17, 18, 19, 23, 24, 25, 27, 28],
    'hb': [6, 10, 11, 12, 13, 14, 18, 24, 29],
  }[dataset_name]

  # T-LESS includes two types of object models, CAD and reconstructed.
  # Use the CAD models as default.
  if dataset_name == 'tless' and model_type is None:
    model_type = 'cad'

  # Name of the folder with object models.
  models_folder_name = 'models'
  if model_type is not None:
    models_folder_name += '_' + model_type

  # Path to the folder with object models.
  models_path = join(datasets_path, dataset_name, models_folder_name)

  p = {
    # ID's of all objects included in the dataset.
    'obj_ids': obj_ids,

    # ID's of objects with symmetries.
    'symmetric_obj_ids': symmetric_obj_ids,

    # Path template to an object model file.
    'model_tpath': join(models_path, 'obj_{obj_id:06d}.ply'),

    # Path to a file with meta information about the object models.
    'models_info_path': join(models_path, 'models_info.json')
  }

  return p


def get_split_params(datasets_path, dataset_name, split, split_type=None):
  """Return parameters (camera params, paths etc.) for the specified dataset.

  :param datasets_path: Path to a folder with datasets.
  :param dataset_name: Name of the dataset for which to return the parameters.
  :param split: Name of the dataset split ('train', 'val', 'test').
  :param split_type: Name of the split type (e.g. for T-LESS, possible types of
    the 'train' split are: 'primesense', 'render_reconst').
  :return: Dictionary with parameters for the specified dataset split.
  """
  p = {
    'name': dataset_name,
    'split': split,
    'split_type': split_type,
    'base_path': join(datasets_path, dataset_name),

    'depth_range': None,
    'azimuth_range': None,
    'elev_range': None,
  }

  gray_ext = '.png'
  rgb_ext = '.png'
  depth_ext = '.png'

  p['im_modalities'] = ['rgb', 'depth']

  # Linemod (LM).
  if dataset_name == 'lm':
    p['scene_ids'] = range(1, 16)
    p['im_size'] = (640, 480)

    if split == 'test':
      p['depth_range'] = (600.90, 1102.35)
      p['azimuth_range'] = (0, 2 * math.pi)
      p['elev_range'] = (0, 0.5 * math.pi)

  # Linemod-Occluded (LM).
  elif dataset_name == 'lmo':
    p['scene_ids'] = {'train': [1, 5, 6, 8, 9, 10, 11, 12], 'test': [2]}[split]
    p['im_size'] = (640, 480)

    if split == 'test':
      p['depth_range'] = (346.31, 1499.84)
      p['azimuth_range'] = (0, 2 * math.pi)
      p['elev_range'] = (0, 0.5 * math.pi)

  # T-LESS.
  elif dataset_name == 'tless':
    p['scene_ids'] = {'train': range(1, 31), 'test': range(1, 21)}[split]

    # Use images from the Primesense sensor by default.
    if split_type is None:
      split_type = 'primesense'

    p['im_size'] = {
      'train': {
        'primesense': (400, 400),
        'kinect': (400, 400),
        'canon': (1900, 1900),
        'render_reconst': (1280, 1024)
      },
      'test': {
        'primesense': (720, 540),
        'kinect': (720, 540),
        'canon': (2560, 1920)
      }
    }[split][split_type]

    # The following holds for Primesense, but is similar for the other sensors.
    if split == 'test':
      p['depth_range'] = (649.89, 940.04)
      p['azimuth_range'] = (0, 2 * math.pi)
      p['elev_range'] = (-0.5 * math.pi, 0.5 * math.pi)

  # TU Dresden Light (TUD-L).
  elif dataset_name == 'tudl':
    if split == 'train' and split_type is None:
      split_type = 'render'

    p['scene_ids'] = range(1, 4)
    p['im_size'] = (640, 480)

    if split == 'test':
      p['depth_range'] = (851.29, 2016.14)
      p['azimuth_range'] = (0, 2 * math.pi)
      p['elev_range'] = (-0.4363, 0.5 * math.pi)  # (-25, 90) [deg].

  # Toyota Light (TYO-L).
  elif dataset_name == 'tyol':
    p['scene_ids'] = range(1, 22)
    p['im_size'] = (640, 480)

    if split == 'test':
      p['depth_range'] = (499.57, 1246.07)
      p['azimuth_range'] = (0, 2 * math.pi)
      p['elev_range'] = (-0.5 * math.pi, 0.5 * math.pi)

  # Rutgers APC (RU-APC).
  elif dataset_name == 'ruapc':
    p['scene_ids'] = range(1, 15)
    p['im_size'] = (640, 480)

    if split == 'test':
      p['depth_range'] = (594.41, 739.12)
      p['azimuth_range'] = (0, 2 * math.pi)
      p['elev_range'] = (-0.5 * math.pi, 0.5 * math.pi)

  # Tejani et al. (IC-MI).
  elif dataset_name == 'icmi':
    p['scene_ids'] = range(1, 7)
    p['im_size'] = (640, 480)

    if split == 'test':
      p['depth_range'] = (509.12, 1120.41)
      p['azimuth_range'] = (0, 2 * math.pi)
      p['elev_range'] = (0, 0.5 * math.pi)

  # Doumanoglou et al. (IC-BIN).
  elif dataset_name == 'icbin':
    p['scene_ids'] = {'train': range(1, 3), 'test': range(1, 4)}[split]
    p['im_size'] = (640, 480)

    if split == 'test':
      p['depth_range'] = (454.56, 1076.29)
      p['azimuth_range'] = (0, 2 * math.pi)
      p['elev_range'] = (-1.0297, 0.5 * math.pi)  # (-59, 90) [deg].

  # MVTec ITODD.
  elif dataset_name == 'itodd':
    p['scene_ids'] = {
      'train': [],
      'val': [1],
      'test': [1]
    }[split]
    p['im_size'] = (1280, 960)

    gray_ext = '.tif'
    depth_ext = '.tif'
    p['im_modalities'] = ['gray', 'depth']

    if split == 'test':
      p['depth_range'] = None
      p['azimuth_range'] = None
      p['elev_range'] = None

  # HomebrewedDB (HB).
  elif dataset_name == 'hb':
    p['scene_ids'] = {
      'train': [],
      'val': [3, 5, 13],
      'test': [3, 5, 13],
    }[split]
    p['im_size'] = (640, 480)

    if split == 'test':
      p['depth_range'] = (420.0, 1430.0)
      p['azimuth_range'] = (0, 2 * math.pi)
      p['elev_range'] = (0.1920, 1.5184)  # (11, 87) [deg].

  else:
    raise ValueError('Unknown BOP dataset.')

  base_path = join(datasets_path, dataset_name)
  split_path = join(base_path, split)
  if split_type is not None:
    split_path += '_' + split_type

  p.update({
    # Path template to a file with per-image camera parameters.
    'scene_camera_tpath': join(
      split_path, '{scene_id:06d}', 'scene_camera.json'),

    # Path template to a file with GT annotations.
    'scene_gt_tpath': join(
      split_path, '{scene_id:06d}', 'scene_gt.json'),

    # Path template to a file with meta information about the GT annotations.
    'scene_gt_info_tpath': join(
      split_path, '{scene_id:06d}', 'scene_gt_info.json'),

    # Path template to a gray image.
    'gray_tpath': join(
      split_path, '{scene_id:06d}', 'gray', '{im_id:06d}' + gray_ext),

    # Path template to an RGB image.
    'rgb_tpath': join(
      split_path, '{scene_id:06d}', 'rgb', '{im_id:06d}' + rgb_ext),

    # Path template to a depth image.
    'depth_tpath': join(
      split_path, '{scene_id:06d}', 'depth', '{im_id:06d}' + depth_ext),

    # Path template to a mask of the full object silhouette.
    'mask_tpath': join(
      split_path, '{scene_id:06d}', 'mask', '{im_id:06d}_{gt_id:06d}.png'),

    # Path template to a mask of the visible part of an object silhouette.
    'mask_visib_tpath': join(
      split_path, '{scene_id:06d}', 'mask_visib',
      '{im_id:06d}_{gt_id:06d}.png'),
  })

  return p