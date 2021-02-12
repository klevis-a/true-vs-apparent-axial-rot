"""Compute true axial rotation based on difference scapula anatomical coordinate systems.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
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
    import numpy as np
    import matplotlib.pyplot as plt
    from true_vs_apparent.common.plot_utils import (init_graphing, mean_sd_plot, style_axes)
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db, extract_sub_rot_norm
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compute true axial rotation based on difference scapula anatomical "
                                     "coordinate systems",
                                     __package__, __file__))
    params = get_params(config_dir / 'parameters.json')

    if not bool(distutils.util.strtobool(os.getenv('VARS_RETAINED', 'False'))):
        # ready db
        db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject, include_anthro=True)
        db['age_group'] = db['Age'].map(lambda age: '<35' if age < 40 else '>45')
        exc_trials = ["O45_003_CA_t01", "O45_003_SA_t02", "O45_003_FE_t02", "U35_010_FE_t01"]
        db = db[~db['Trial_Name'].str.contains('|'.join(exc_trials))]
        db['Trial'].apply(pre_fetch)

    # relevant parameters
    output_path = Path(params.output_dir)

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    db_elev_gc = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    db_elev_ac = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    db_elev_aa = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev_gc, params.torso_def, 'GC', params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    prepare_db(db_elev_ac, params.torso_def, 'AC', params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    prepare_db(db_elev_aa, params.torso_def, 'PLA', params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])

    #%%
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', 'd']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev_gc.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig_diff = plt.figure(figsize=(90 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs_diff = fig_diff.subplots(3, 1)

    # style axes, add x and y labels
    style_axes(axs_diff[0], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff[1], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff[2], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')

    # plot
    max_pos = 140
    leg_mean = []
    gc_groupby = db_elev_gc.groupby('Activity', observed=True)
    ac_groupby = db_elev_ac.groupby('Activity', observed=True)
    aa_groupby = db_elev_aa.groupby('Activity', observed=True)
    for idx, (act_gc, act_df_gc) in enumerate(gc_groupby):
        act_df_ac = ac_groupby.get_group(act_gc)
        act_df_aa = aa_groupby.get_group(act_gc)

        all_traj_true_gc = np.stack(act_df_gc['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)
        all_traj_true_ac = np.stack(act_df_ac['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)
        all_traj_true_aa = np.stack(act_df_aa['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)

        # means and standard deviations
        true_mean_gc = np.rad2deg(np.mean(all_traj_true_gc, axis=0))
        true_mean_ac = np.rad2deg(np.mean(all_traj_true_ac, axis=0))
        true_mean_aa = np.rad2deg(np.mean(all_traj_true_aa, axis=0))
        true_sd_gc = np.rad2deg(np.std(all_traj_true_gc, ddof=1, axis=0))
        true_sd_ac = np.rad2deg(np.std(all_traj_true_ac, ddof=1, axis=0))
        true_sd_aa = np.rad2deg(np.std(all_traj_true_aa, ddof=1, axis=0))

        # plot mean +- sd
        cur_row = act_row[act_gc.lower()]
        true_ln_gc = mean_sd_plot(axs_diff[cur_row], x, true_mean_gc, true_sd_gc,
                                  dict(color=color_map.colors[0], alpha=0.3),
                                  dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        true_ln_ac = mean_sd_plot(axs_diff[cur_row], x, true_mean_ac, true_sd_ac,
                                  dict(color=color_map.colors[1], alpha=0.3),
                                  dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        true_ln_aa = mean_sd_plot(axs_diff[cur_row], x, true_mean_aa, true_sd_aa,
                                  dict(color=color_map.colors[2], alpha=0.3),
                                  dict(color=color_map.colors[2], marker=markers[2], markevery=20))

        if idx == 0:
            leg_mean.append(true_ln_gc[0])
            leg_mean.append(true_ln_ac[0])
            leg_mean.append(true_ln_aa[0])

    # figure title and legend
    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
    fig_diff.suptitle('Effect of scapula anatomical CS\non true GH axial rotation', x=0.56, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.91)
    leg = fig_diff.legend(leg_mean, ['GC', 'AC', 'AA'], loc='upper left',
                          bbox_to_anchor=(0, 1), ncol=1, handlelength=1.5, handletextpad=0.5,
                          columnspacing=0.75, borderpad=0.2)

    # set x ticks
    if int(params.min_elev) % 10 == 0:
        x_ticks_start = int(params.min_elev)
    else:
        x_ticks_start = int(params.min_elev) - int(params.min_elev) % 10

    if int(params.max_elev) % 10 == 0:
        x_ticks_end = int(params.max_elev)
    else:
        x_ticks_end = int(params.max_elev) + (10 - int(params.max_elev) % 10)
    x_ticks = np.arange(x_ticks_start, x_ticks_end + 1, 20)
    x_ticks = np.sort(np.concatenate((x_ticks, np.array([params.min_elev, params.max_elev]))))
    for ax in axs_diff:
        ax.set_xticks(x_ticks)

    # add axes titles
    _, y0, _, h = axs_diff[0].get_position().bounds
    fig_diff.text(0.5, y0 + h * 1.03, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff[1].get_position().bounds
    fig_diff.text(0.5, y0 + h * 1.03, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff[2].get_position().bounds
    fig_diff.text(0.5, y0 + h * 1.03, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    # add arrows indicating direction
    axs_diff[0].arrow(35, 45, 0, -20, length_includes_head=True, head_width=2, head_length=2)
    axs_diff[0].text(23, 45, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    if params.fig_file:
        fig_diff.savefig(params.fig_file)
    else:
        plt.show()
