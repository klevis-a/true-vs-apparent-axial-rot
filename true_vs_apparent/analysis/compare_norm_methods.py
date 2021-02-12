"""Compare normalizing by first frame of trial versus start of always increasing HT elevation.

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
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from true_vs_apparent.common.plot_utils import init_graphing, make_interactive, mean_sd_plot, style_axes
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db, extract_sub_rot_norm
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('DO NOT USE', __package__, __file__))
    params = get_params(config_dir / 'parameters.json')

    if not bool(distutils.util.strtobool(os.getenv('Compare normalizing by first frame of trial versus start of always '
                                                   'increasing HT elevation', 'False'))):
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
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig_norm_hum = plt.figure(figsize=(270 / 25.4, 190 / 25.4))
    axs_norm = fig_norm_hum.subplots(3, 3)

    # style axes, add x and y labels
    style_axes(axs_norm[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_norm[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_norm[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_norm[0, 1], None, None)
    style_axes(axs_norm[1, 1], None, None)
    style_axes(axs_norm[2, 1], 'Humerothoracic Elevation (Deg)', None)
    style_axes(axs_norm[0, 2], None, None)
    style_axes(axs_norm[1, 2], None, None)
    style_axes(axs_norm[2, 2], 'Humerothoracic Elevation (Deg)', None)

    # plot
    leg_patch_mean = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_isb_norm0 = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'traj']), axis=0)
        all_traj_phadke_norm0 = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'traj']), axis=0)
        all_traj_true_norm0 = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'traj']), axis=0)

        all_traj_isb_normup = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_phadke_normup = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true_normup = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)

        # means and standard deviations
        isb_mean_norm0 = np.rad2deg(np.mean(all_traj_isb_norm0, axis=0))
        phadke_mean_norm0 = np.rad2deg(np.mean(all_traj_phadke_norm0, axis=0))
        true_mean_norm0 = np.rad2deg(np.mean(all_traj_true_norm0, axis=0))
        isb_sd_norm0 = np.rad2deg(np.std(all_traj_isb_norm0, ddof=1, axis=0))
        phadke_sd_norm0 = np.rad2deg(np.std(all_traj_phadke_norm0, ddof=1, axis=0))
        true_sd_norm0 = np.rad2deg(np.std(all_traj_true_norm0, ddof=1, axis=0))

        isb_mean_normup = np.rad2deg(np.mean(all_traj_isb_normup, axis=0))
        phadke_mean_normup = np.rad2deg(np.mean(all_traj_phadke_normup, axis=0))
        true_mean_normup = np.rad2deg(np.mean(all_traj_true_normup, axis=0))
        isb_sd_normup = np.rad2deg(np.std(all_traj_isb_normup, ddof=1, axis=0))
        phadke_sd_normup = np.rad2deg(np.std(all_traj_phadke_normup, ddof=1, axis=0))
        true_sd_normup = np.rad2deg(np.std(all_traj_true_normup, ddof=1, axis=0))

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        isb_ln_norm0 = mean_sd_plot(axs_norm[cur_row, 0], x, isb_mean_norm0, isb_sd_norm0,
                                    dict(color=color_map.colors[0], alpha=0.3),
                                    dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        phadke_ln_norm0 = mean_sd_plot(axs_norm[cur_row, 1], x, phadke_mean_norm0, phadke_sd_norm0,
                                       dict(color=color_map.colors[2], alpha=0.25),
                                       dict(color=color_map.colors[2], marker=markers[1], markevery=20))
        true_ln_norm0 = mean_sd_plot(axs_norm[cur_row, 2], x, true_mean_norm0, true_sd_norm0,
                                     dict(color=color_map.colors[4], alpha=0.25),
                                     dict(color=color_map.colors[4], marker=markers[2], markevery=20))

        isb_ln_normup = mean_sd_plot(axs_norm[cur_row, 0], x, isb_mean_normup, isb_sd_norm0,
                                     dict(color=color_map.colors[1], alpha=0.3),
                                     dict(color=color_map.colors[1], marker=markers[0], markevery=20,
                                          ls='--', fillstyle='none'))
        phadke_ln_normup = mean_sd_plot(axs_norm[cur_row, 1], x, phadke_mean_normup, phadke_sd_norm0,
                                        dict(color=color_map.colors[3], alpha=0.25),
                                        dict(color=color_map.colors[3], marker=markers[1], markevery=20,
                                             ls='--', fillstyle='none'))
        true_ln_normup = mean_sd_plot(axs_norm[cur_row, 2], x, true_mean_normup, true_sd_norm0,
                                      dict(color=color_map.colors[5], alpha=0.25),
                                      dict(color=color_map.colors[5], marker=markers[2], markevery=20,
                                           ls='--', fillstyle='none'))

        if idx == 0:
            leg_patch_mean.append(isb_ln_norm0[0])
            leg_patch_mean.append(isb_ln_normup[0])
            leg_patch_mean.append(phadke_ln_norm0[0])
            leg_patch_mean.append(phadke_ln_normup[0])
            leg_patch_mean.append(true_ln_norm0[0])
            leg_patch_mean.append(true_ln_normup[0])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_norm_hum.suptitle('Glenohumeral Motion Comparison', x=0.3, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_norm_hum.legend(
        leg_patch_mean, ['ISB', 'ISB', "xz'y''", "xz'y''", 'True Axial From Zero', 'True Axial From Start'],
        loc='upper right', bbox_to_anchor=(1.01, 1.01), ncol=3, handlelength=1.5, handletextpad=0.5, columnspacing=0.75)

    # add axes titles
    _, y0, _, h = axs_norm[0, 0].get_position().bounds
    fig_norm_hum.text(0.5, y0 + h * 1.05, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_norm[1, 0].get_position().bounds
    fig_norm_hum.text(0.5, y0 + h * 1.05, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_norm[2, 0].get_position().bounds
    fig_norm_hum.text(0.5, y0 + h * 1.05, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()
    plt.show()
