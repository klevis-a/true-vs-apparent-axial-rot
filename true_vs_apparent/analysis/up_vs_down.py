"""Compare yx'y'', zx'y'' and true GH axial rotation for differences between arm raising and lowering.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
use_ac: Whether to use the AC or GC landmark when building the scapula CS.
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

    config_dir = Path(mod_arg_parser("Compare yx'y'', zx'y'' and true GH axial rotation for differences between arm "
                                     "raising and lowering", __package__, __file__))
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
    use_ac = bool(distutils.util.strtobool(params.use_ac))

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, use_ac, params.dtheta_fine, params.dtheta_coarse,
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
        fig_updown = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
        axs_updown = fig_updown.subplots(3, 2)

        # style axes, add x and y labels
        style_axes(axs_updown[0, 0], None, 'Axial Rotation (Deg)')
        style_axes(axs_updown[1, 0], None, 'Axial Rotation (Deg)')
        style_axes(axs_updown[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
        style_axes(axs_updown[0, 1], None, 'SPM{t}')
        style_axes(axs_updown[1, 1], None, 'SPM{t}')
        style_axes(axs_updown[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{t}')

        all_traj_isb_up = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_isb', 2]), axis=0)
        all_traj_phadke_up = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2]), axis=0)
        all_traj_true_up = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'true_axial_rot', None]), axis=0)
        all_traj_isb_up = all_traj_isb_up - all_traj_isb_up[:, 0][..., np.newaxis]
        all_traj_phadke_up = all_traj_phadke_up - all_traj_phadke_up[:, 0][..., np.newaxis]
        all_traj_true_up = all_traj_true_up - all_traj_true_up[:, 0][..., np.newaxis]

        all_traj_isb_down = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_down', 'euler.gh_isb', 2]), axis=0)
        all_traj_phadke_down = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_down', 'euler.gh_phadke', 2]), axis=0)
        all_traj_true_down = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_down', 'true_axial_rot', None]), axis=0)
        all_traj_isb_down = all_traj_isb_down - all_traj_isb_down[:, 0][..., np.newaxis]
        all_traj_phadke_down = all_traj_phadke_down - all_traj_phadke_down[:, 0][..., np.newaxis]
        all_traj_true_down = all_traj_true_down - all_traj_true_down[:, 0][..., np.newaxis]

        # means and standard deviations
        isb_mean_up = np.rad2deg(np.mean(all_traj_isb_up, axis=0))
        phadke_mean_up = np.rad2deg(np.mean(all_traj_phadke_up, axis=0))
        true_mean_up = np.rad2deg(np.mean(all_traj_true_up, axis=0))
        isb_sd_up = np.rad2deg(np.std(all_traj_isb_up, ddof=1, axis=0))
        phadke_sd_up = np.rad2deg(np.std(all_traj_phadke_up, ddof=1, axis=0))
        true_sd_up = np.rad2deg(np.std(all_traj_true_up, ddof=1, axis=0))

        isb_mean_down = np.rad2deg(np.mean(all_traj_isb_down, axis=0))
        phadke_mean_down = np.rad2deg(np.mean(all_traj_phadke_down, axis=0))
        true_mean_down = np.rad2deg(np.mean(all_traj_true_down, axis=0))
        isb_sd_down = np.rad2deg(np.std(all_traj_isb_down, ddof=1, axis=0))
        phadke_sd_down = np.rad2deg(np.std(all_traj_phadke_down, ddof=1, axis=0))
        true_sd_down = np.rad2deg(np.std(all_traj_true_down, ddof=1, axis=0))

        # spm
        up_v_down_isb = spm1d.stats.ttest_paired(all_traj_isb_up[:, 1:],
                                                 all_traj_isb_down[:, 1:]).inference(alpha, two_tailed=True)
        up_v_down_phadke = spm1d.stats.ttest_paired(all_traj_phadke_up[:, 1:],
                                                    all_traj_phadke_down[:, 1:]).inference(alpha, two_tailed=True)
        up_v_down_true = spm1d.stats.ttest_paired(all_traj_true_up[:, 1:],
                                                  all_traj_true_down[:, 1:]).inference(alpha, two_tailed=True)

        # plot mean +- sd
        shaded_up = dict(color=color_map.colors[0], alpha=0.25)
        shaded_down = dict(color=color_map.colors[1], alpha=0.25)
        line_up = dict(color=color_map.colors[0], marker=markers[0], markevery=20)
        line_down = dict(color=color_map.colors[1], marker=markers[1], markevery=20)
        up_true_ln = mean_sd_plot(axs_updown[0, 0], x, true_mean_up, true_sd_up, shaded_up, line_up)
        down_true_ln = mean_sd_plot(axs_updown[0, 0], x, true_mean_down, true_sd_down, shaded_down, line_down)
        up_isb_ln = mean_sd_plot(axs_updown[1, 0], x, isb_mean_up, isb_sd_up, shaded_up, line_up)
        down_isb_ln = mean_sd_plot(axs_updown[1, 0], x, isb_mean_down, isb_sd_down, shaded_down, line_down)
        up_phadke_ln = mean_sd_plot(axs_updown[2, 0], x, phadke_mean_up, phadke_sd_up, shaded_up, line_up)
        down_phadke_ln = mean_sd_plot(axs_updown[2, 0], x, phadke_mean_down, phadke_sd_down, shaded_down, line_down)

        # plot spm
        spm_shaded = dict(color=color_map.colors[2], alpha=0.25)
        spm_line = dict(color=color_map.colors[2])
        true_spm_ln = spm_plot(axs_updown[0, 1], x[1:], up_v_down_true, spm_shaded, spm_line)
        isb_spm_ln = spm_plot(axs_updown[1, 1], x[1:], up_v_down_isb, spm_shaded, spm_line)
        phadke_spm_ln = spm_plot(axs_updown[2, 1], x[1:], up_v_down_phadke, spm_shaded, spm_line)

        # figure title and legend
        plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
        fig_updown.suptitle(activity + ' Glenohumeral Up vs Down Axial Rotation Comparison', x=0.5, y=0.99,
                            fontweight='bold')
        plt.subplots_adjust(top=0.93)
        axs_updown[0, 0].legend([up_true_ln[0], down_true_ln[0]], ['Up', 'Down'], loc='upper left',
                                bbox_to_anchor=(0, 1.05), ncol=2, handlelength=1.5, handletextpad=0.5,
                                columnspacing=0.75)
        axs_updown[0, 1].legend([true_spm_ln[0]], ['SPM {t}'], loc='lower left', bbox_to_anchor=(0, 0.1), ncol=1,
                                handlelength=1.5, handletextpad=0.5, columnspacing=0.75)

        # add axes titles
        _, y0, _, h = axs_updown[0, 0].get_position().bounds
        fig_updown.text(0.5, y0 + h * 1.05, 'True Axial Rotation', ha='center', fontsize=11, fontweight='bold')

        _, y0, _, h = axs_updown[1, 0].get_position().bounds
        fig_updown.text(0.5, y0 + h * 1.05, 'ISB Axial Rotation', ha='center', fontsize=11, fontweight='bold')

        _, y0, _, h = axs_updown[2, 0].get_position().bounds
        fig_updown.text(0.5, y0 + h * 1.05, 'Phadke Axial rotation', ha='center', fontsize=11, fontweight='bold')

        make_interactive()

    # same as above but normalized by axial rotation at start
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        fig_updown = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
        axs_updown = fig_updown.subplots(3, 2)

        # style axes, add x and y labels
        style_axes(axs_updown[0, 0], None, 'Axial Rotation (Deg)')
        style_axes(axs_updown[1, 0], None, 'Axial Rotation (Deg)')
        style_axes(axs_updown[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
        style_axes(axs_updown[0, 1], None, 'SPM{t}')
        style_axes(axs_updown[1, 1], None, 'SPM{t}')
        style_axes(axs_updown[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{t}')

        all_traj_isb_up = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_phadke_up = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true_up = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)

        all_traj_isb_down = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_down', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_phadke_down = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_down', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true_down = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_down', 'true_axial_rot', None, 'up']), axis=0)

        # means and standard deviations
        isb_mean_up = np.rad2deg(np.mean(all_traj_isb_up, axis=0))
        phadke_mean_up = np.rad2deg(np.mean(all_traj_phadke_up, axis=0))
        true_mean_up = np.rad2deg(np.mean(all_traj_true_up, axis=0))
        isb_sd_up = np.rad2deg(np.std(all_traj_isb_up, ddof=1, axis=0))
        phadke_sd_up = np.rad2deg(np.std(all_traj_phadke_up, ddof=1, axis=0))
        true_sd_up = np.rad2deg(np.std(all_traj_true_up, ddof=1, axis=0))

        isb_mean_down = np.rad2deg(np.mean(all_traj_isb_down, axis=0))
        phadke_mean_down = np.rad2deg(np.mean(all_traj_phadke_down, axis=0))
        true_mean_down = np.rad2deg(np.mean(all_traj_true_down, axis=0))
        isb_sd_down = np.rad2deg(np.std(all_traj_isb_down, ddof=1, axis=0))
        phadke_sd_down = np.rad2deg(np.std(all_traj_phadke_down, ddof=1, axis=0))
        true_sd_down = np.rad2deg(np.std(all_traj_true_down, ddof=1, axis=0))

        # spm
        up_v_down_isb = spm1d.stats.ttest_paired(all_traj_isb_up, all_traj_isb_down).inference(alpha, two_tailed=True)
        up_v_down_phadke = spm1d.stats.ttest_paired(all_traj_phadke_up, all_traj_phadke_down).inference(alpha,
                                                                                                        two_tailed=True)
        up_v_down_true = spm1d.stats.ttest_paired(all_traj_true_up, all_traj_true_down).inference(alpha,
                                                                                                  two_tailed=True)

        # plot mean +- sd
        shaded_up = dict(color=color_map.colors[0], alpha=0.25)
        shaded_down = dict(color=color_map.colors[1], alpha=0.25)
        line_up = dict(color=color_map.colors[0], marker=markers[0], markevery=20)
        line_down = dict(color=color_map.colors[1], marker=markers[1], markevery=20)
        up_true_ln = mean_sd_plot(axs_updown[0, 0], x, true_mean_up, true_sd_up, shaded_up, line_up)
        down_true_ln = mean_sd_plot(axs_updown[0, 0], x, true_mean_down, true_sd_down, shaded_down, line_down)
        up_isb_ln = mean_sd_plot(axs_updown[1, 0], x, isb_mean_up, isb_sd_up, shaded_up, line_up)
        down_isb_ln = mean_sd_plot(axs_updown[1, 0], x, isb_mean_down, isb_sd_down, shaded_down, line_down)
        up_phadke_ln = mean_sd_plot(axs_updown[2, 0], x, phadke_mean_up, phadke_sd_up, shaded_up, line_up)
        down_phadke_ln = mean_sd_plot(axs_updown[2, 0], x, phadke_mean_down, phadke_sd_down, shaded_down, line_down)

        # plot spm
        spm_shaded = dict(color=color_map.colors[2], alpha=0.25)
        spm_line = dict(color=color_map.colors[2])
        true_spm_ln = spm_plot(axs_updown[0, 1], x, up_v_down_true, spm_shaded, spm_line)
        isb_spm_ln = spm_plot(axs_updown[1, 1], x, up_v_down_isb, spm_shaded, spm_line)
        phadke_spm_ln = spm_plot(axs_updown[2, 1], x, up_v_down_phadke, spm_shaded, spm_line)

        # figure title and legend
        plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
        fig_updown.suptitle(activity + ' GH Up vs Down Axial Rotation Comparison Norm Start', x=0.5, y=0.99,
                            fontweight='bold')
        plt.subplots_adjust(top=0.93)
        axs_updown[0, 0].legend([up_true_ln[0], down_true_ln[0]], ['Up', 'Down'], loc='upper left',
                                bbox_to_anchor=(0, 1.05), ncol=2, handlelength=1.5, handletextpad=0.5,
                                columnspacing=0.75)
        axs_updown[0, 1].legend([true_spm_ln[0]], ['SPM {t}'], loc='lower left', bbox_to_anchor=(0, 0.1), ncol=1,
                                handlelength=1.5, handletextpad=0.5, columnspacing=0.75)

        # add axes titles
        _, y0, _, h = axs_updown[0, 0].get_position().bounds
        fig_updown.text(0.5, y0 + h * 1.05, 'True Axial Rotation', ha='center', fontsize=11, fontweight='bold')

        _, y0, _, h = axs_updown[1, 0].get_position().bounds
        fig_updown.text(0.5, y0 + h * 1.05, 'ISB Axial Rotation', ha='center', fontsize=11, fontweight='bold')

        _, y0, _, h = axs_updown[2, 0].get_position().bounds
        fig_updown.text(0.5, y0 + h * 1.05, 'Phadke Axial rotation', ha='center', fontsize=11, fontweight='bold')

        make_interactive()
    plt.show()
