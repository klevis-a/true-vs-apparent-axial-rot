"""Compare yx'y'', zx'y'' and true GH axial rotation for differences between age groups.

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
    from true_vs_apparent.common.plot_utils import init_graphing, make_interactive, mean_sd_plot, spm_plot, style_axes
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db, extract_sub_rot, extract_sub_rot_norm
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare yx'y'', zx'y'' and true GH axial rotation for differences between age "
                                     "groups", __package__, __file__))
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
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    # plot
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        fig_age = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
        axs_age = fig_age.subplots(3, 2)

        # style axes, add x and y labels
        style_axes(axs_age[0, 0], None, 'Axial Rotation (Deg)')
        style_axes(axs_age[1, 0], None, 'Axial Rotation (Deg)')
        style_axes(axs_age[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
        style_axes(axs_age[0, 1], None, 'SPM{t}')
        style_axes(axs_age[1, 1], None, 'SPM{t}')
        style_axes(axs_age[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{t}')

        activity_df_lt35 = activity_df.loc[activity_df['age_group'] == '<35']
        activity_df_gt45 = activity_df.loc[activity_df['age_group'] == '>45']

        all_traj_isb_lt35 = np.stack(activity_df_lt35['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_isb', 2]), axis=0)
        all_traj_phadke_lt35 = np.stack(activity_df_lt35['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2]), axis=0)
        all_traj_true_lt35 = np.stack(activity_df_lt35['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'true_axial_rot', None]), axis=0)
        all_traj_isb_lt35 = all_traj_isb_lt35 - all_traj_isb_lt35[:, 0][..., np.newaxis]
        all_traj_phadke_lt35 = all_traj_phadke_lt35 - all_traj_phadke_lt35[:, 0][..., np.newaxis]
        all_traj_true_lt35 = all_traj_true_lt35 - all_traj_true_lt35[:, 0][..., np.newaxis]

        all_traj_isb_gt45 = np.stack(activity_df_gt45['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_isb', 2]), axis=0)
        all_traj_phadke_gt45 = np.stack(activity_df_gt45['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2]), axis=0)
        all_traj_true_gt45 = np.stack(activity_df_gt45['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'true_axial_rot', None]), axis=0)
        all_traj_isb_gt45 = all_traj_isb_gt45 - all_traj_isb_gt45[:, 0][..., np.newaxis]
        all_traj_phadke_gt45 = all_traj_phadke_gt45 - all_traj_phadke_gt45[:, 0][..., np.newaxis]
        all_traj_true_gt45 = all_traj_true_gt45 - all_traj_true_gt45[:, 0][..., np.newaxis]

        # means and standard deviations
        isb_mean_lt35 = np.rad2deg(np.mean(all_traj_isb_lt35, axis=0))
        phadke_mean_lt35 = np.rad2deg(np.mean(all_traj_phadke_lt35, axis=0))
        true_mean_lt35 = np.rad2deg(np.mean(all_traj_true_lt35, axis=0))
        isb_sd_lt35 = np.rad2deg(np.std(all_traj_isb_lt35, ddof=1, axis=0))
        phadke_sd_lt35 = np.rad2deg(np.std(all_traj_phadke_lt35, ddof=1, axis=0))
        true_sd_lt35 = np.rad2deg(np.std(all_traj_true_lt35, ddof=1, axis=0))

        isb_mean_gt45 = np.rad2deg(np.mean(all_traj_isb_gt45, axis=0))
        phadke_mean_gt45 = np.rad2deg(np.mean(all_traj_phadke_gt45, axis=0))
        true_mean_gt45 = np.rad2deg(np.mean(all_traj_true_gt45, axis=0))
        isb_sd_gt45 = np.rad2deg(np.std(all_traj_isb_gt45, ddof=1, axis=0))
        phadke_sd_gt45 = np.rad2deg(np.std(all_traj_phadke_gt45, ddof=1, axis=0))
        true_sd_gt45 = np.rad2deg(np.std(all_traj_true_gt45, ddof=1, axis=0))

        # spm
        lt35_v_gt45_isb = spm1d.stats.ttest2(all_traj_isb_lt35[:, 1:], all_traj_isb_gt45[:, 1:],
                                             equal_var=False).inference(alpha, two_tailed=True)
        lt35_v_gt45_phadke = spm1d.stats.ttest2(all_traj_phadke_lt35[:, 1:], all_traj_phadke_gt45[:, 1:],
                                                equal_var=False).inference(alpha, two_tailed=True)
        lt35_v_gt45_true = spm1d.stats.ttest2(all_traj_true_lt35[:, 1:], all_traj_true_gt45[:, 1:],
                                              equal_var=False).inference(alpha, two_tailed=True)

        # plot mean +- sd
        shaded_lt35 = dict(color=color_map.colors[0], alpha=0.25)
        shaded_gt45 = dict(color=color_map.colors[1], alpha=0.25)
        line_lt35 = dict(color=color_map.colors[0], marker=markers[0], markevery=20)
        line_gt45 = dict(color=color_map.colors[1], marker=markers[1], markevery=20)
        lt35_true_ln = mean_sd_plot(axs_age[0, 0], x, true_mean_lt35, true_sd_lt35, shaded_lt35, line_lt35)
        gt45_true_ln = mean_sd_plot(axs_age[0, 0], x, true_mean_gt45, true_sd_gt45, shaded_gt45, line_gt45)
        lt35_isb_ln = mean_sd_plot(axs_age[1, 0], x, isb_mean_lt35, isb_sd_lt35, shaded_lt35, line_lt35)
        gt45_isb_ln = mean_sd_plot(axs_age[1, 0], x, isb_mean_gt45, isb_sd_gt45, shaded_gt45, line_gt45)
        lt35_phadke_ln = mean_sd_plot(axs_age[2, 0], x, phadke_mean_lt35, phadke_sd_lt35, shaded_lt35, line_lt35)
        gt45_phadke_ln = mean_sd_plot(axs_age[2, 0], x, phadke_mean_gt45, phadke_sd_gt45, shaded_gt45, line_gt45)

        # plot spm
        spm_shaded = dict(color=color_map.colors[2], alpha=0.25)
        spm_line = dict(color=color_map.colors[2])
        true_spm_ln = spm_plot(axs_age[0, 1], x[1:], lt35_v_gt45_true, spm_shaded, spm_line)
        isb_spm_ln = spm_plot(axs_age[1, 1], x[1:], lt35_v_gt45_isb, spm_shaded, spm_line)
        phadke_spm_ln = spm_plot(axs_age[2, 1], x[1:], lt35_v_gt45_phadke, spm_shaded, spm_line)

        # figure title and legend
        plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
        fig_age.suptitle(activity + ' Glenohumeral Axial Rotation Comparison', x=0.5, y=0.99, fontweight='bold')
        plt.subplots_adjust(top=0.93)
        axs_age[0, 0].legend([lt35_true_ln[0], gt45_true_ln[0]], ['Less Than 35', 'Greater Than 45'], loc='upper left',
                             bbox_to_anchor=(0, 1.05), ncol=2, handlelength=1.5, handletextpad=0.5,
                             columnspacing=0.75)
        axs_age[0, 1].legend([true_spm_ln[0]], ['SPM {t}'], loc='lower left', bbox_to_anchor=(0, 0.1), ncol=1,
                             handlelength=1.5, handletextpad=0.5, columnspacing=0.75)

        # add axes titles
        _, y0, _, h = axs_age[0, 0].get_position().bounds
        fig_age.text(0.5, y0 + h * 1.05, 'True Axial Rotation', ha='center', fontsize=11, fontweight='bold')

        _, y0, _, h = axs_age[1, 0].get_position().bounds
        fig_age.text(0.5, y0 + h * 1.05, 'ISB Axial Rotation', ha='center', fontsize=11, fontweight='bold')

        _, y0, _, h = axs_age[2, 0].get_position().bounds
        fig_age.text(0.5, y0 + h * 1.05, 'Phadke Axial rotation', ha='center', fontsize=11, fontweight='bold')

        make_interactive()

    # same as above but normalized by start of motion
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        fig_age = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
        axs_age = fig_age.subplots(3, 2)

        # style axes, add x and y labels
        style_axes(axs_age[0, 0], None, 'Axial Rotation (Deg)')
        style_axes(axs_age[1, 0], None, 'Axial Rotation (Deg)')
        style_axes(axs_age[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
        style_axes(axs_age[0, 1], None, 'SPM{t}')
        style_axes(axs_age[1, 1], None, 'SPM{t}')
        style_axes(axs_age[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{t}')

        activity_df_lt35 = activity_df.loc[activity_df['age_group'] == '<35']
        activity_df_gt45 = activity_df.loc[activity_df['age_group'] == '>45']

        all_traj_isb_lt35 = np.stack(activity_df_lt35['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_phadke_lt35 = np.stack(activity_df_lt35['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true_lt35 = np.stack(activity_df_lt35['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)

        all_traj_isb_gt45 = np.stack(activity_df_gt45['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_phadke_gt45 = np.stack(activity_df_gt45['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true_gt45 = np.stack(activity_df_gt45['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)

        # means and standard deviations
        isb_mean_lt35 = np.rad2deg(np.mean(all_traj_isb_lt35, axis=0))
        phadke_mean_lt35 = np.rad2deg(np.mean(all_traj_phadke_lt35, axis=0))
        true_mean_lt35 = np.rad2deg(np.mean(all_traj_true_lt35, axis=0))
        isb_sd_lt35 = np.rad2deg(np.std(all_traj_isb_lt35, ddof=1, axis=0))
        phadke_sd_lt35 = np.rad2deg(np.std(all_traj_phadke_lt35, ddof=1, axis=0))
        true_sd_lt35 = np.rad2deg(np.std(all_traj_true_lt35, ddof=1, axis=0))

        isb_mean_gt45 = np.rad2deg(np.mean(all_traj_isb_gt45, axis=0))
        phadke_mean_gt45 = np.rad2deg(np.mean(all_traj_phadke_gt45, axis=0))
        true_mean_gt45 = np.rad2deg(np.mean(all_traj_true_gt45, axis=0))
        isb_sd_gt45 = np.rad2deg(np.std(all_traj_isb_gt45, ddof=1, axis=0))
        phadke_sd_gt45 = np.rad2deg(np.std(all_traj_phadke_gt45, ddof=1, axis=0))
        true_sd_gt45 = np.rad2deg(np.std(all_traj_true_gt45, ddof=1, axis=0))

        # spm
        lt35_v_gt45_isb = spm1d.stats.ttest2(all_traj_isb_lt35[:, 1:], all_traj_isb_gt45[:, 1:],
                                             equal_var=False).inference(alpha, two_tailed=True)
        lt35_v_gt45_phadke = spm1d.stats.ttest2(all_traj_phadke_lt35[:, 1:], all_traj_phadke_gt45[:, 1:],
                                                equal_var=False).inference(alpha, two_tailed=True)
        lt35_v_gt45_true = spm1d.stats.ttest2(all_traj_true_lt35[:, 1:], all_traj_true_gt45[:, 1:],
                                              equal_var=False).inference(alpha, two_tailed=True)

        # plot mean +- sd
        shaded_lt35 = dict(color=color_map.colors[0], alpha=0.25)
        shaded_gt45 = dict(color=color_map.colors[1], alpha=0.25)
        line_lt35 = dict(color=color_map.colors[0], marker=markers[0], markevery=20)
        line_gt45 = dict(color=color_map.colors[1], marker=markers[1], markevery=20)
        lt35_true_ln = mean_sd_plot(axs_age[0, 0], x, true_mean_lt35, true_sd_lt35, shaded_lt35, line_lt35)
        gt45_true_ln = mean_sd_plot(axs_age[0, 0], x, true_mean_gt45, true_sd_gt45, shaded_gt45, line_gt45)
        lt35_isb_ln = mean_sd_plot(axs_age[1, 0], x, isb_mean_lt35, isb_sd_lt35, shaded_lt35, line_lt35)
        gt45_isb_ln = mean_sd_plot(axs_age[1, 0], x, isb_mean_gt45, isb_sd_gt45, shaded_gt45, line_gt45)
        lt35_phadke_ln = mean_sd_plot(axs_age[2, 0], x, phadke_mean_lt35, phadke_sd_lt35, shaded_lt35, line_lt35)
        gt45_phadke_ln = mean_sd_plot(axs_age[2, 0], x, phadke_mean_gt45, phadke_sd_gt45, shaded_gt45, line_gt45)

        # plot spm
        spm_shaded = dict(color=color_map.colors[2], alpha=0.25)
        spm_line = dict(color=color_map.colors[2])
        true_spm_ln = spm_plot(axs_age[0, 1], x[1:], lt35_v_gt45_true, spm_shaded, spm_line)
        isb_spm_ln = spm_plot(axs_age[1, 1], x[1:], lt35_v_gt45_isb, spm_shaded, spm_line)
        phadke_spm_ln = spm_plot(axs_age[2, 1], x[1:], lt35_v_gt45_phadke, spm_shaded, spm_line)

        # figure title and legend
        plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
        fig_age.suptitle(activity + ' Glenohumeral Axial Rotation Comparison Norm Start', x=0.5, y=0.99,
                         fontweight='bold')
        plt.subplots_adjust(top=0.93)
        axs_age[0, 0].legend([lt35_true_ln[0], gt45_true_ln[0]], ['Less Than 35', 'Greater Than 45'], loc='upper left',
                             bbox_to_anchor=(0, 1.05), ncol=2, handlelength=1.5, handletextpad=0.5,
                             columnspacing=0.75)
        axs_age[0, 1].legend([true_spm_ln[0]], ['SPM {t}'], loc='lower left', bbox_to_anchor=(0, 0.1), ncol=1,
                             handlelength=1.5, handletextpad=0.5, columnspacing=0.75)

        # add axes titles
        _, y0, _, h = axs_age[0, 0].get_position().bounds
        fig_age.text(0.5, y0 + h * 1.05, 'True Axial Rotation', ha='center', fontsize=11, fontweight='bold')

        _, y0, _, h = axs_age[1, 0].get_position().bounds
        fig_age.text(0.5, y0 + h * 1.05, 'ISB Axial Rotation', ha='center', fontsize=11, fontweight='bold')

        _, y0, _, h = axs_age[2, 0].get_position().bounds
        fig_age.text(0.5, y0 + h * 1.05, 'Phadke Axial rotation', ha='center', fontsize=11, fontweight='bold')

        make_interactive()
    plt.show()
