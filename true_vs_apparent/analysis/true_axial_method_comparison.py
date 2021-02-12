"""Compare different methods of computing true GH and HT axial rotation

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
scap_lateral: Landmarks to utilize when defining the scapula's lateral (+Z) axis (AC, PLA, GC).
torso_def: Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition.
dtheta_fine: Incremental angle (deg) to use for fine interpolation between minimum and maximum HT elevation analyzed.
dtheta_coarse: Incremental angle (deg) to use for coarse interpolation between minimum and maximum HT elevation analyzed.
min_elev: Minimum HT elevation angle (deg) utilized for analysis that encompasses all trials.
max_elev: Maximum HT elevation angle (deg) utilized for analysis that encompasses all trials.
"""
import numpy as np
import quaternion as q
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R
from biokinepy.vec_ops import extended_dot


def quat_project(quat, axis):
    axis_project = np.dot(q.as_float_array(quat)[1:], axis) * axis
    quat_proj = np.quaternion(q.as_float_array(quat)[0], axis_project[0], axis_project[1], axis_project[2])
    return quat_proj / np.absolute(quat_proj)


def true_axial_rot_ang_vel(traj):
    ang_vel_proj = extended_dot(traj.ang_vel, traj.rot_matrix[:, :, 1])
    return cumtrapz(ang_vel_proj, dx=traj.dt, initial=0)


def true_axial_rot_fha(traj):
    num_frames = traj.rot_matrix.shape[0]
    traj_diff = traj.quat[1:] * np.conjugate(traj.quat[:-1])
    # Switch to using scipy.spatial.transform here because the quaternion package gives a rotation of close to 2*pi
    # sometimes. This isn't incorrect, but it's not consistent. The return should be 2*pi-ret
    # scipy.spatial.transform uses a scalar-last convention so account for that here
    traj_diff_float = q.as_float_array(traj_diff)
    traj_diff_float = np.concatenate((traj_diff_float[:, 1:], traj_diff_float[:, 0][..., np.newaxis]), axis=1)
    traj_fha = R.from_quat(traj_diff_float).as_rotvec()
    traj_fha_proj = extended_dot(traj_fha, traj.rot_matrix[:-1, :, 1])
    true_axial_rot = np.empty((num_frames,), dtype=np.float)
    true_axial_rot[0] = 0
    true_axial_rot[1:] = np.add.accumulate(traj_fha_proj)
    return true_axial_rot


def true_axial_rot_swing_twist(traj):
    num_frames = traj.rot_matrix.shape[0]
    # rotational difference between frames expressed in torso coordinate system
    traj_diff = traj.quat[1:] * np.conjugate(traj.quat[:-1])

    true_axial_rot_delta = np.empty((num_frames,), dtype=np.float)
    true_axial_rot_delta[0] = 0
    for i in range(num_frames-1):
        hum_axis = traj.rot_matrix[i, :, 1]
        diff_proj = q.as_float_array(quat_project(traj_diff[i], hum_axis))
        rot_vec = R.from_quat(np.array([diff_proj[1], diff_proj[2], diff_proj[3], diff_proj[0]])).as_rotvec()
        rot_vec_theta = np.linalg.norm(rot_vec)
        rot_vec_axis = rot_vec / rot_vec_theta
        # Note that rot_vec_theta will always be + because of np.linalg.norm. But a rotation about an axis v by an angle
        # theta is the same as a rotation about -v by an angle -theta. So here the humeral axis sets our direction. That
        # is, we always rotate around hum_axis (and not -hum_axis) and adjust the sign of rot_vec_theta accordingly
        true_axial_rot_delta[i+1] = rot_vec_theta * (1 if np.dot(rot_vec_axis, hum_axis) > 0 else -1)

    return np.add.accumulate(true_axial_rot_delta)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db
    from true_vs_apparent.common.analysis_er_utils import ready_er_db
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare different methods of computing true GH and HT axial rotation',
                                     __package__, __file__))
    params = get_params(config_dir / 'parameters.json')

    if not bool(distutils.util.strtobool(os.getenv('VARS_RETAINED', 'False'))):
        # ready db
        db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject, include_anthro=True)
        db['age_group'] = db['Age'].map(lambda age: '<35' if age < 40 else '>45')
        if params.excluded_trials:
            db = db[~db['Trial_Name'].str.contains('|'.join(params.excluded_trials))]
        db['Trial'].apply(pre_fetch)

    # relevant parameters
    output_path = Path(params.output_dir)

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # prepare db
    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    db_rot = db.loc[db['Trial_Name'].str.contains('_ERa90_|_ERaR_')].copy()
    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_rot = ready_er_db(db_rot, params.torso_def, params.scap_lateral, params.erar_endpts, params.era90_endpts,
                         params.dtheta_fine)

    # compute true axial rotation with each of the three different methods
    db_elev['ang_vel'] = db_elev[params.trajectory].apply(true_axial_rot_ang_vel)
    db_elev['fha'] = db_elev[params.trajectory].apply(true_axial_rot_fha)
    db_elev['swing_twist'] = db_elev[params.trajectory].apply(true_axial_rot_swing_twist)
    db_rot['ang_vel'] = db_rot[params.trajectory].apply(true_axial_rot_ang_vel)
    db_rot['fha'] = db_rot[params.trajectory].apply(true_axial_rot_fha)
    db_rot['swing_twist'] = db_rot[params.trajectory].apply(true_axial_rot_swing_twist)

    # compare differences
    db_elev['ang_vel_vs_fha'] = db_elev['ang_vel'] - db_elev['fha']
    db_elev['ang_vel_vs_swing_twist'] = db_elev['ang_vel'] - db_elev['swing_twist']
    db_elev['fha_vs_swing_twist'] = db_elev['fha'] - db_elev['swing_twist']
    db_rot['ang_vel_vs_fha'] = db_rot['ang_vel'] - db_rot['fha']
    db_rot['ang_vel_vs_swing_twist'] = db_rot['ang_vel'] - db_rot['swing_twist']
    db_rot['fha_vs_swing_twist'] = db_rot['fha'] - db_rot['swing_twist']

    db_elev['ang_vel_vs_fha_max'] = \
        db_elev['ang_vel_vs_fha'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_elev['ang_vel_vs_swing_twist_max'] = \
        db_elev['ang_vel_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_elev['fha_vs_swing_twist_max'] = \
        db_elev['fha_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_rot['ang_vel_vs_fha_max'] = \
        db_rot['ang_vel_vs_fha'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_rot['ang_vel_vs_swing_twist_max'] = \
        db_rot['ang_vel_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_rot['fha_vs_swing_twist_max'] = \
        db_rot['fha_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)

    print('ELEVATION')
    print('Angular Velocity vs FHA Maximum Difference: {:.2f}'.format(db_elev['ang_vel_vs_fha_max'].max()))
    print('Angular Velocity vs Swing Twist Maximum Difference: {:.2f}'.format(db_elev['ang_vel_vs_swing_twist_max']
                                                                              .max()))
    print('FHA vs Swing Twist Maximum Difference: {:.2f}'.format(db_elev['fha_vs_swing_twist_max'].max()))

    print('INTERNAL/EXTERNAL ROTATION')
    print('Angular Velocity vs FHA Maximum Difference: {:.2f}'.format(db_rot['ang_vel_vs_fha_max'].max()))
    print('Angular Velocity vs Swing Twist Maximum Difference: {:.2f}'.format(db_rot['ang_vel_vs_swing_twist_max']
                                                                              .max()))
    print('FHA vs Swing Twist Maximum Difference: {:.2f}'.format(db_rot['fha_vs_swing_twist_max'].max()))
