"""Compare yx'y'', zx'y'' and true GH axial rotation against zero.

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
parametric: Whether to use a parametric (true) or non-parametric statistical test (false).
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
    from true_vs_apparent.common.plot_utils import init_graphing, make_interactive, mean_sd_plot, spm_plot, style_axes
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db, extract_sub_rot, extract_sub_rot_norm
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare yx'y'', zx'y'' and true GH axial rotation against zero",
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
        spm_test = spm1d.stats.ttest
        infer_params = {}
    else:
        spm_test = spm1d.stats.nonparam.ttest
        infer_params = {"force_iterations": True}
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig_diff0_hum = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_diff0 = fig_diff0_hum.subplots(3, 2)

    # style axes, add x and y labels
    style_axes(axs_diff0[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff0[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff0[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_diff0[0, 1], None, 'SPM{t}')
    style_axes(axs_diff0[1, 1], None, 'SPM{t}')
    style_axes(axs_diff0[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{t}')

    fig_diff0_dist = plt.figure(figsize=(110 / 25.4, 190 / 25.4))
    axs_diff0_dist = fig_diff0_dist.subplots(3, 1)
    style_axes(axs_diff0_dist[0], None, 'p-value')
    style_axes(axs_diff0_dist[1], None, 'p-value')
    style_axes(axs_diff0_dist[2], 'Humerothoracic Elevation (Deg)', 'p-value')

    # plot
    leg_patch_mean = []
    leg_patch_dist = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_isb = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_isb', 2]), axis=0)
        all_traj_phadke = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2]), axis=0)
        all_traj_true = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'true_axial_rot', None]), axis=0)
        all_traj_isb = all_traj_isb - all_traj_isb[:, 0][..., np.newaxis]
        all_traj_phadke = all_traj_phadke - all_traj_phadke[:, 0][..., np.newaxis]
        all_traj_true = all_traj_true - all_traj_true[:, 0][..., np.newaxis]

        # means and standard deviations
        isb_mean = np.rad2deg(np.mean(all_traj_isb, axis=0))
        phadke_mean = np.rad2deg(np.mean(all_traj_phadke, axis=0))
        true_mean = np.rad2deg(np.mean(all_traj_true, axis=0))
        isb_sd = np.rad2deg(np.std(all_traj_isb, ddof=1, axis=0))
        phadke_sd = np.rad2deg(np.std(all_traj_phadke, ddof=1, axis=0))
        true_sd = np.rad2deg(np.std(all_traj_true, ddof=1, axis=0))

        # spm
        isb_zero = spm_test(all_traj_isb[:, 1:], 0).inference(alpha, two_tailed=True, **infer_params)
        isb_zero_norm = spm1d.stats.normality.sw.ttest(all_traj_isb[:, 1:])
        phadke_zero = spm_test(all_traj_phadke[:, 1:], 0).inference(alpha, two_tailed=True, **infer_params)
        phadke_zero_norm = spm1d.stats.normality.sw.ttest(all_traj_phadke[:, 1:])
        true_zero = spm_test(all_traj_true[:, 1:], 0).inference(alpha, two_tailed=True, **infer_params)
        true_zero_norm = spm1d.stats.normality.sw.ttest(all_traj_true[:, 1:])

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        true_ln = mean_sd_plot(axs_diff0[cur_row, 0], x, true_mean, true_sd,
                               dict(color=color_map.colors[2], alpha=0.25),
                               dict(color=color_map.colors[2], marker=markers[2], markevery=20))
        phadke_ln = mean_sd_plot(axs_diff0[cur_row, 0], x, phadke_mean, phadke_sd,
                                 dict(color=color_map.colors[1], alpha=0.25),
                                 dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        isb_ln = mean_sd_plot(axs_diff0[cur_row, 0], x, isb_mean, isb_sd, dict(color=color_map.colors[0], alpha=0.3),
                              dict(color=color_map.colors[0], marker=markers[0], markevery=20))

        # plot spm
        isb_t_ln = spm_plot(axs_diff0[cur_row, 1], x[1:], isb_zero, dict(color=color_map.colors[0], alpha=0.25),
                            dict(color=color_map.colors[0]))
        phadke_t_ln = spm_plot(axs_diff0[cur_row, 1], x[1:], phadke_zero, dict(color=color_map.colors[1], alpha=0.25),
                               dict(color=color_map.colors[1]))
        true_t_ln = spm_plot(axs_diff0[cur_row, 1], x[1:], true_zero, dict(color=color_map.colors[2], alpha=0.25),
                             dict(color=color_map.colors[2]))

        axs_diff0_dist[cur_row].axhline(0.05, ls='--')
        isb_dist_ln = axs_diff0_dist[cur_row].plot(x[1:], isb_zero_norm[1], color=color_map.colors[0])
        phadke_dist_ln = axs_diff0_dist[cur_row].plot(x[1:], phadke_zero_norm[1], color=color_map.colors[1])
        true_dist_ln = axs_diff0_dist[cur_row].plot(x[1:], true_zero_norm[1], color=color_map.colors[2])

        leg_patch_dist.append(isb_dist_ln[0])
        leg_patch_dist.append(phadke_dist_ln[0])
        leg_patch_dist.append(true_dist_ln[0])

        if idx == 0:
            leg_patch_mean.append(isb_ln[0])
            leg_patch_mean.append(isb_t_ln[0])
            leg_patch_mean.append(phadke_ln[0])
            leg_patch_mean.append(phadke_t_ln[0])
            leg_patch_mean.append(true_ln[0])
            leg_patch_mean.append(true_t_ln[0])

    # figure title and legend
    plt.figure(fig_diff0_dist.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    plt.figure(fig_diff0_hum.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_diff0_hum.suptitle('Glenohumeral Motion Comparison', x=0.3, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_diff0_hum.legend(
        leg_patch_mean, ['ISB', 'ISB', "xz'y''", "xz'y''", 'True Axial Mean', 'True Axial SPM{t}'],
        loc='upper right', bbox_to_anchor=(1.01, 1.01), ncol=3, handlelength=1.5, handletextpad=0.5, columnspacing=0.75)

    plt.figure(fig_diff0_dist.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_diff0_dist.suptitle('Normality', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_diff0_dist.legend(leg_patch_dist, ['ISB', 'Phadke'], loc='upper right')

    # add axes titles
    _, y0, _, h = axs_diff0[0, 0].get_position().bounds
    fig_diff0_hum.text(0.5, y0 + h * 1.05, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff0[1, 0].get_position().bounds
    fig_diff0_hum.text(0.5, y0 + h * 1.05, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff0[2, 0].get_position().bounds
    fig_diff0_hum.text(0.5, y0 + h * 1.05, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    # #### Same comparison as above but the rotation is normalized by the starting orientation #####
    fig_diff0_norm = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_diff0_norm = fig_diff0_norm.subplots(3, 2)

    # style axes, add x and y labels
    style_axes(axs_diff0_norm[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff0_norm[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff0_norm[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_diff0_norm[0, 1], None, 'SPM{t}')
    style_axes(axs_diff0_norm[1, 1], None, 'SPM{t}')
    style_axes(axs_diff0_norm[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{t}')

    fig_diff0_norm_dist = plt.figure(figsize=(110 / 25.4, 190 / 25.4))
    axs_diff0_norm_dist = fig_diff0_norm_dist.subplots(3, 1)
    style_axes(axs_diff0_norm_dist[0], None, 'p-value')
    style_axes(axs_diff0_norm_dist[1], None, 'p-value')
    style_axes(axs_diff0_norm_dist[2], 'Humerothoracic Elevation (Deg)', 'p-value')

    # plot
    leg_patch_mean_norm = []
    leg_patch_mean_norm_dist = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_isb = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_phadke = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)

        # means and standard deviations
        isb_mean = np.rad2deg(np.mean(all_traj_isb, axis=0))
        phadke_mean = np.rad2deg(np.mean(all_traj_phadke, axis=0))
        true_mean = np.rad2deg(np.mean(all_traj_true, axis=0))
        isb_sd = np.rad2deg(np.std(all_traj_isb, ddof=1, axis=0))
        phadke_sd = np.rad2deg(np.std(all_traj_phadke, ddof=1, axis=0))
        true_sd = np.rad2deg(np.std(all_traj_true, ddof=1, axis=0))

        # spm
        isb_zero = spm_test(all_traj_isb, 0).inference(alpha, two_tailed=True, **infer_params)
        isb_zero_norm = spm1d.stats.normality.sw.ttest(all_traj_isb)
        phadke_zero = spm_test(all_traj_phadke, 0).inference(alpha, two_tailed=True, **infer_params)
        phadke_zero_norm = spm1d.stats.normality.sw.ttest(all_traj_phadke)
        true_zero = spm_test(all_traj_true, 0).inference(alpha, two_tailed=True, **infer_params)
        true_zero_norm = spm1d.stats.normality.sw.ttest(all_traj_true)

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        true_ln = mean_sd_plot(axs_diff0_norm[cur_row, 0], x, true_mean, true_sd,
                               dict(color=color_map.colors[2], alpha=0.25),
                               dict(color=color_map.colors[2], marker=markers[2], markevery=20))
        phadke_ln = mean_sd_plot(axs_diff0_norm[cur_row, 0], x, phadke_mean, phadke_sd,
                                 dict(color=color_map.colors[1], alpha=0.25),
                                 dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        isb_ln = mean_sd_plot(axs_diff0_norm[cur_row, 0], x, isb_mean, isb_sd,
                              dict(color=color_map.colors[0], alpha=0.3),
                              dict(color=color_map.colors[0], marker=markers[0], markevery=20))

        # plot spm
        isb_t_ln = spm_plot(axs_diff0_norm[cur_row, 1], x, isb_zero, dict(color=color_map.colors[0], alpha=0.25),
                            dict(color=color_map.colors[0]))
        phadke_t_ln = spm_plot(axs_diff0_norm[cur_row, 1], x, phadke_zero, dict(color=color_map.colors[1], alpha=0.25),
                               dict(color=color_map.colors[1]))
        true_t_ln = spm_plot(axs_diff0_norm[cur_row, 1], x, true_zero, dict(color=color_map.colors[2], alpha=0.25),
                             dict(color=color_map.colors[2]))

        axs_diff0_norm_dist[cur_row].axhline(-0.05, ls='--')
        axs_diff0_norm_dist[cur_row].axhline(0.05, ls='--')
        isb_norm_dist_ln = axs_diff0_norm_dist[cur_row].plot(x, isb_zero_norm[1], color=color_map.colors[0])
        phadke_norm_dist_ln = axs_diff0_norm_dist[cur_row].plot(x, phadke_zero_norm[1], color=color_map.colors[1])
        true_norm_dist_ln = axs_diff0_norm_dist[cur_row].plot(x, true_zero_norm[1], color=color_map.colors[2])

        leg_patch_mean_norm_dist.append(isb_norm_dist_ln[0])
        leg_patch_mean_norm_dist.append(phadke_norm_dist_ln[0])
        leg_patch_mean_norm_dist.append(true_norm_dist_ln[0])

        if idx == 0:
            leg_patch_mean_norm.append(isb_ln[0])
            leg_patch_mean_norm.append(isb_t_ln[0])
            leg_patch_mean_norm.append(phadke_ln[0])
            leg_patch_mean_norm.append(phadke_t_ln[0])
            leg_patch_mean_norm.append(true_ln[0])
            leg_patch_mean_norm.append(true_t_ln[0])

    # figure title and legend
    plt.figure(fig_diff0_norm_dist.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    plt.figure(fig_diff0_norm.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_diff0_norm.suptitle('GH Motion Comparison Norm Start', x=0.3, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_diff0_norm.legend(
        leg_patch_mean_norm, ['ISB', 'ISB', "xz'y''", "xz'y''", 'True Axial Mean', 'True Axial SPM{t}'],
        loc='upper right', bbox_to_anchor=(1.01, 1.01), ncol=3, handlelength=1.5, handletextpad=0.5, columnspacing=0.75)

    plt.figure(fig_diff0_norm_dist.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_diff0_norm_dist.suptitle('Normality', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_diff0_norm_dist.legend(leg_patch_mean_norm_dist, ['ISB', 'Phadke'], loc='upper right')

    # add axes titles
    _, y0, _, h = axs_diff0_norm[0, 0].get_position().bounds
    fig_diff0_norm.text(0.5, y0 + h * 1.05, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff0_norm[1, 0].get_position().bounds
    fig_diff0_norm.text(0.5, y0 + h * 1.05, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff0_norm[2, 0].get_position().bounds
    fig_diff0_norm.text(0.5, y0 + h * 1.05, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    plt.show()
