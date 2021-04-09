"""Determine the drift due to linearization

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
era90_endpts: Path to csv file containing start and stop frames (including both external and internal rotation) for
external rotation in 90 deg of abduction trials.
erar_endpts: Path to csv file containing start and stop frames (including both external and internal rotation) for
external rotation in adduction trials.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from true_vs_apparent.common.analysis_er_utils import ready_er_db


def compute_drift(gh, dt):
    num_frames = gh.rot_matrix.shape[0]
    computed_orients = np.zeros((num_frames, 3, 3))
    angle_diff = np.zeros((num_frames, ))
    for i in range(num_frames):
        if i == 0:
            computed_orients[0] = np.eye(3)
            angle_diff[0] = 0
        else:
            computed_orient = Rotation.from_rotvec(gh.ang_vel[i-1] * dt).as_matrix() @ gh.rot_matrix[i-1]
            compute_diff = gh.rot_matrix[i].T @ computed_orient
            compute_rotvec = Rotation.from_matrix(compute_diff).as_rotvec()
            angle_diff[i] = np.rad2deg(np.sqrt(np.dot(compute_rotvec, compute_rotvec)))

    return angle_diff


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Determine the drift due to linearization', __package__, __file__))
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

    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_er = ready_er_db(db, params.torso_def, 'GC', params.erar_endpts, params.era90_endpts,
                        params.dtheta_fine)
    db_elev['drift_traj'] = db_elev['gh'].apply(compute_drift, args=[db.attrs['dt']])
    db_elev['drift_max'] = db_elev['drift_traj'].apply(np.max)
    db_er['drift_traj'] = db_er['gh'].apply(compute_drift, args=[db.attrs['dt']])
    db_er['drift_max'] = db_er['drift_traj'].apply(np.max)

    all_drift_max = np.concatenate((db_elev['drift_max'].to_numpy(), db_er['drift_max'].to_numpy()))
    print('Min: {:.2f} Max: {:.2f} Median: {:.2f} SD: {:.2f} 25th: {:.2f} 75th: {:.2f}'.
          format(np.min(all_drift_max), np.max(all_drift_max), np.median(all_drift_max), np.std(all_drift_max),
                 *np.quantile(all_drift_max, [0.25, 0.75])))
