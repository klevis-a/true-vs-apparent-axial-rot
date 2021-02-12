"""Visualize how accounting for plane of elevation affects the ISB decomposition

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
    import spm1d
    from true_vs_apparent.common.plot_utils import init_graphing, make_interactive, mean_sd_plot, style_axes
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db, extract_sub_rot_norm, sub_rot_at_max_elev
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Visualize how accounting for plane of elevation affects the ISB decomposition",
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

    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])

    #%%
    if bool(distutils.util.strtobool(params.parametric)):
        spm_test = spm1d.stats.ttest_paired
        infer_params = {}
    else:
        spm_test = spm1d.stats.nonparam.ttest_paired
        infer_params = {'force_iterations': True}

    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig_diff_hum = plt.figure(figsize=(100 / 25.4, 190 / 25.4))
    axs_diff = fig_diff_hum.subplots(3, 1)

    # style axes, add x and y labels
    style_axes(axs_diff[0], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff[1], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff[2], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')

    # plot
    leg_patch_mean = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_isb = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_isb_poe = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 0, 'up']), axis=0)
        all_traj_true = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)
        all_traj_isb_norm = all_traj_isb + all_traj_isb_poe

        all_traj_isb_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_isb_poe_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'euler.gh_isb', 0, 'up']), axis=0)
        all_traj_true_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'true_axial_rot', None, 'up']), axis=0)
        all_traj_isb_norm_max = all_traj_isb_max + all_traj_isb_poe_max

        # means and standard deviations
        isb_mean = np.rad2deg(np.mean(all_traj_isb, axis=0))
        true_mean = np.rad2deg(np.mean(all_traj_true, axis=0))
        isb_norm_mean = np.rad2deg(np.mean(all_traj_isb_norm, axis=0))
        isb_sd = np.rad2deg(np.std(all_traj_isb, ddof=1, axis=0))
        true_sd = np.rad2deg(np.std(all_traj_true, ddof=1, axis=0))
        isb_norm_sd = np.rad2deg(np.std(all_traj_isb_norm, ddof=1, axis=0))

        isb_mean_max = np.rad2deg(np.mean(all_traj_isb_max, axis=0))
        true_mean_max = np.rad2deg(np.mean(all_traj_true_max, axis=0))
        isb_norm_mean_max = np.rad2deg(np.mean(all_traj_isb_norm_max, axis=0))
        isb_sd_max = np.rad2deg(np.std(all_traj_isb_max, ddof=1, axis=0))
        true_sd_max = np.rad2deg(np.std(all_traj_true_max, ddof=1, axis=0))
        isb_norm_sd_max = np.rad2deg(np.std(all_traj_isb_norm_max, ddof=1, axis=0))

        isb_poe_mean = np.rad2deg(np.mean(all_traj_isb_poe, axis=0))
        isb_poe_sd = np.rad2deg(np.std(all_traj_isb_poe, ddof=1, axis=0))
        isb_poe_mean_max = np.rad2deg(np.mean(all_traj_isb_poe_max, axis=0))
        isb_poe_sd_max = np.rad2deg(np.std(all_traj_isb_poe_max, ddof=1, axis=0))

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        true_ln = mean_sd_plot(axs_diff[cur_row], x, true_mean, true_sd,
                               dict(color=color_map.colors[2], alpha=0.25),
                               dict(color=color_map.colors[2], marker=markers[2], markevery=20))
        isb_norm_ln = mean_sd_plot(axs_diff[cur_row], x, isb_norm_mean, isb_norm_sd,
                                   dict(color=color_map.colors[1], alpha=0.25),
                                   dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        isb_ln = mean_sd_plot(axs_diff[cur_row], x, isb_mean, isb_sd,
                              dict(color=color_map.colors[0], alpha=0.3),
                              dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        # poe_ln = mean_sd_plot(axs_diff[cur_row], x, isb_poe_mean, isb_poe_sd,
        #                       dict(color=color_map.colors[3], alpha=0.3),
        #                       dict(color=color_map.colors[3], marker=markers[0], markevery=20))

        # plot endpoint
        axs_diff[cur_row].errorbar(x[-1] + (x[-1] - x[0]) * 0.1, true_mean_max, yerr=true_sd_max,
                                   color=color_map.colors[2], marker=markers[2], capsize=3)
        axs_diff[cur_row].errorbar(x[-1] + (x[-1] - x[0]) * 0.14, isb_norm_mean_max, yerr=isb_norm_sd_max,
                                   color=color_map.colors[1], marker=markers[1], capsize=3)
        axs_diff[cur_row].errorbar(x[-1] + (x[-1] - x[0]) * 0.12, isb_mean_max, yerr=isb_sd_max,
                                   color=color_map.colors[0], marker=markers[0], capsize=3)
        # axs_diff[cur_row].errorbar(x[-1] + (x[-1] - x[0]) * 0.16, isb_poe_mean_max, yerr=isb_poe_sd_max,
        #                            color=color_map.colors[3], marker=markers[3], capsize=3)

        if idx == 0:
            leg_patch_mean.append(isb_ln[0])
            leg_patch_mean.append(isb_norm_ln[0])
            leg_patch_mean.append(true_ln[0])

    # figure title and legend
    plt.figure(fig_diff_hum.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_diff_hum.suptitle('Glenohumeral Motion Comparison', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    axs_diff[0].legend(leg_patch_mean, ['ISB', "ISB Adj", 'True Axial Mean'], loc='upper left',
                       bbox_to_anchor=(0, 1.05), ncol=3, handlelength=1.5, handletextpad=0.5, columnspacing=0.75)

    # add axes titles
    _, y0, _, h = axs_diff[0].get_position().bounds
    fig_diff_hum.text(0.5, y0 + h * 1.05, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff[1].get_position().bounds
    fig_diff_hum.text(0.5, y0 + h * 1.05, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff[2].get_position().bounds
    fig_diff_hum.text(0.5, y0 + h * 1.05, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()
    plt.show()
