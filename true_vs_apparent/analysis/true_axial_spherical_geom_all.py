"""Verify that angular difference between sequence-specified trajectory and the true trajectory is the spherical area
between the two trajectories.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
use_ac: Whether to use the AC or GC landmark when building the scapula CS.
decomp_method: Decomposition method to verify (isb, phadke).
trajectory: gh or th.
"""

from spherical_geometry.polygon import SphericalPolygon
from true_vs_apparent.common.python_utils import rgetattr


def phadke_path(orient, start_idx, end_idx):
    # compile longitudes and latitudes for computing difference in axial rotation
    long = []
    lat = []

    # add starting orientation
    long.append(orient[start_idx, 0])
    lat.append(orient[start_idx, 1])

    # from starting orientation come back to zero latitude
    long.append(orient[start_idx, 0])
    lat.append(0)

    # keep zero latitude and go to final longitude
    long.append(orient[end_idx, 0])
    lat.append(0)

    # now work backwards back via actual path of humerus
    for i in range(end_idx, start_idx - 1, -1):
        long.append(orient[i, 0])
        lat.append(orient[i, 1])

    return long, lat


def isb_path(orient, start_idx, end_idx):
    # compile longitudes and latitudes for computing difference in axial rotation
    long = []
    lat = []

    # add starting orientation
    # we want zero latitude to be the equator but -90 deg of elevation is the equator so fix by adding 90 deg
    long.append(orient[start_idx, 0])
    lat.append(orient[start_idx, 1] + 90)

    # go to zero
    long.append(0)
    lat.append(0 + 90)

    # now work backwards back via actual path of humerus
    for i in range(end_idx, start_idx - 1, -1):
        long.append(orient[i, 0])
        lat.append(orient[i, 1] + 90)

    return long, lat


def phadke_add_axial_diff(orient, start_idx, end_idx):
    return 0


def isb_add_axial_diff(orient, start_idx, end_idx):
    # for isb there is axial rotation that happens when establishing PoE
    return orient[end_idx, 0] - orient[start_idx, 0]


def true_axial_analysis(df_row, traj_def, euler_def, path_fnc, add_axial_rot_fnc):
    traj = df_row[traj_def]
    start_idx = df_row['up_down_analysis'].max_run_up_start_idx
    end_idx = df_row['up_down_analysis'].max_run_up_end_idx
    orient = np.rad2deg(rgetattr(traj, euler_def))
    true_axial = np.rad2deg(getattr(traj, 'true_axial_rot'))
    apparent_orient_diff = orient[end_idx, 2] - orient[start_idx, 2]
    true_axial_diff = true_axial[end_idx] - true_axial[start_idx]
    add_axial_rot = add_axial_rot_fnc(orient, start_idx, end_idx)

    long, lat = path_fnc(orient, start_idx, end_idx)

    # compute the area
    mid_ix = int((start_idx + end_idx) / 2)
    sp = SphericalPolygon.from_lonlat(long, lat, center=(orient[mid_ix, 0], orient[mid_ix, 1]/2))
    area = np.rad2deg(sp.area())

    # if the actual path and the "euler" path cross each other the spherical_geometry polygon incorrectly estimates the
    # area
    while area > 180:
        area -= 180

    return apparent_orient_diff, true_axial_diff, area, add_axial_rot, sp.is_clockwise()


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import numpy as np
    import distutils.util
    from pathlib import Path
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject
    from true_vs_apparent.common.analysis_utils import prepare_db
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    from logging.config import fileConfig
    import logging

    config_dir = Path(mod_arg_parser('Apparent - true axial rotation = spherical area', __package__, __file__))
    params = get_params(config_dir / 'parameters.json')
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)

    # relevant parameters
    use_ac = bool(distutils.util.strtobool(params.use_ac))

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # prepare database
    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, use_ac, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev], should_fill=False, should_clean=False)

    if 'isb' in params.decomp_method:
        fnc_path = isb_path
        fnc_axial_orient_diff = isb_add_axial_diff
        decomp_method = 'euler.yxy_intrinsic'
    elif 'phadke' in params.decomp_method:
        fnc_path = phadke_path
        fnc_axial_orient_diff = phadke_add_axial_diff
        decomp_method = 'euler.xzy_intrinsic'
    else:
        raise ValueError('Only ISB and Phadke comparisons have been implemented.')

    # keep only elevation trials
    (db_elev['apparent_axial_rot'], db_elev['true_axial_rot'], db_elev['sphere_area'],
     db_elev['add_axial_rot'], db_elev['clockwise']) = \
        zip(*db_elev.apply(true_axial_analysis, args=[params.trajectory, decomp_method, fnc_path,
                                                      fnc_axial_orient_diff], axis=1))
    summary_db = db_elev.loc[:, ['apparent_axial_rot', 'true_axial_rot', 'sphere_area', 'add_axial_rot', 'clockwise']]

    # Use absolute values to compare the computed area and the difference between the true axial rotation and the
    # difference between starting and ending axial orientation. Maybe it would be possible to drop the absolute value
    # and do actual comparisons but I would have to get into the weeds of the spherical_geometry package and that
    # doesn't seem necessary since we will be using the true axial rotation as derived from angular velocity anyways.
    # In addition, sometimes the area is reported as 140 deg when it should be 180-140 deg, but sometimes 140 deg
    # is the correct area. So here I use both estimates
    summary_db['sphere_area_abs'] = summary_db['sphere_area'].apply(np.absolute)
    summary_db['180-sphere_area_abs'] = (180 - summary_db['sphere_area']).apply(np.absolute)
    summary_db['true_apparent_axial_diff'] = (summary_db['true_axial_rot'] - summary_db['apparent_axial_rot'] -
                                              summary_db['add_axial_rot']).apply(np.absolute)
    summary_db['delta'] = (summary_db['sphere_area_abs'] - summary_db['true_apparent_axial_diff']).apply(np.absolute)
    summary_db['180-delta'] = (summary_db['180-sphere_area_abs'] -
                               summary_db['true_apparent_axial_diff']).apply(np.absolute)
    summary_db['delta_min'] = summary_db[['delta', '180-delta']].min(axis=1)
