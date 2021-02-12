"""Plot GH yx'y'' axial rotation for individual subjects

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
backend: Matplotlib backend to use for plotting (e.g. Qt5Agg, macosx, etc.).
dpi: Dots (pixels) per inch for generated figure. (e.g. 300)
fig_file: Path to file where to save figure.
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from true_vs_apparent.common.plot_utils import (init_graphing, style_axes, make_interactive)
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    from true_vs_apparent.analysis.overview_elev_trials import summary_plotter, ind_plotter
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Plot GH yx'y'' axial rotation for individual subjects", __package__, __file__))
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

    #%%
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20c.colors)
    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(90 / 25.4, 190 / 25.4), dpi=params.dpi)
    ax = fig.subplots(3, 1)

    for i in range(3):
        style_axes(ax[i], 'Humerothoracic Elevation (Deg)' if i == 2 else None, 'Axial Rotation (Deg)')
        ax[i].xaxis.set_major_locator(ticker.MultipleLocator(base=20.0))

    act_row = {'ca': 0, 'sa': 1, 'fe': 2}
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        cur_row = act_row[activity.lower()]
        _, agg_lines, quat_mean_lines = \
            summary_plotter(activity_df['traj_interp'], 'gh', 'ht_ea_up', 'up', 'common_ht_range_coarse',
                            'common_coarse_up', 'euler.gh_isb', 2, ax[cur_row], ind_plotter, 'black', 'red',
                            error_bars='std', alpha=0.5)
        # remove the quaternion average line
        quat_mean_lines[0].remove()
        del quat_mean_lines[0]
        make_interactive()

    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
    fig.suptitle("GH yx'y'' Axial Orientation", fontsize=12, fontweight='bold', y=0.99)
    plt.subplots_adjust(top=0.93)
    fig.legend([agg_lines[0]], ['Mean'], ncol=1, handlelength=1.00, handletextpad=0.25, columnspacing=0.5,
               loc='upper right', bbox_to_anchor=(1, 1))
    # add axes titles
    _, y0, _, h = ax[0].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = ax[1].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = ax[2].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
