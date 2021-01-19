"""Check the filling of the N003A_SA_t01 trial.

This script checks the filling of the N003A_SA_t01 trial. I elected to (slightly) fill this trial because it was within
1 deg of achieving the minimum range of 25 deg HT elevation, thus allowing us to use this range for all the other
subjects. The filling is based from the linear and angular velocity of the trajectory.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
use_ac: Whether to use the AC or GC landmark when building the scapula CS.
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import numpy as np
    import matplotlib.pyplot as plt
    import distutils.util
    from pathlib import Path
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject
    from true_vs_apparent.common.analysis_utils import prepare_db
    from true_vs_apparent.analysis.up_down_identify import extract_up_down_min_max
    from true_vs_apparent.common import plot_utils
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    from logging.config import fileConfig
    import logging

    config_dir = Path(mod_arg_parser('Check N003A_SA_t01 filling', __package__, __file__))
    params = get_params(config_dir / 'parameters.json')
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)
    db = db.loc[['N003A_SA_t01']]

    # relevant parameters
    use_ac = bool(distutils.util.strtobool(params.use_ac))

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # compute min and max ht elevation for each subject
    prepare_db(db, params.torso_def, use_ac, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev], should_clean=False)
    (db['up_min_ht'], db['up_max_ht'], db['down_min_ht'], db['down_max_ht']) = zip(
        *(db['up_down_analysis'].apply(extract_up_down_min_max)))

    plot_utils.init_graphing(params.backend)
    plot_dirs = [['ht', 'ht_isb', 'HT'], ['gh', 'gh_isb', 'GH'], ['st', 'st_isb', 'ST']]
    for plot_dir in plot_dirs:
        traj = db.loc['N003A_SA_t01', plot_dir[0]]
        traj_euler = getattr(traj, 'euler')
        fig = plt.figure(figsize=(14, 7), tight_layout=True)
        ax = fig.subplots(2, 3)
        for i in range(3):
            ax[0, i].plot(np.rad2deg(getattr(traj_euler, plot_dir[1])[:, i]))
            ax[1, i].plot(traj.pos[:, i])
            if i == 0:
                plot_utils.style_axes(ax[0, i], None, 'Orientation (deg)')
                plot_utils.style_axes(ax[1, i], 'Frame Index (Zero-Based)', 'Position (mm)')
            else:
                plot_utils.style_axes(ax[0, i], None, None)
                plot_utils.style_axes(ax[1, i], 'Frame Index (Zero-Based)', None)
        fig.suptitle(plot_dir[2])
        plot_utils.make_interactive()
    plt.show()