"""Compare HT yx'y'' and zx'y'' GH axial rotation against true axial rotation normalized by their values at 25 deg of
HT elevation.

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
parametric: Whether to use a parametric (true) or non-parametric statistical test (false).
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
    import matplotlib.ticker as ticker
    import spm1d
    from true_vs_apparent.common.plot_utils import (init_graphing, mean_sd_plot, extract_sig, style_axes, sig_filter)
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db, extract_sub_rot_norm, sub_rot_at_max_elev
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare HT yx'y'' and zx'y'' GH axial rotation against true axial rotation "
                                     "normalized by 25 HT elev", __package__, __file__))
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

    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, 'GC', params.dtheta_fine, params.dtheta_coarse,
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
    markers = ['^', 'o',  's', 'd']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig_diff = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs_diff = fig_diff.subplots(3, 2)

    # style axes, add x and y labels
    style_axes(axs_diff[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_diff[0, 1], None, None)
    style_axes(axs_diff[1, 1], None, None)
    style_axes(axs_diff[2, 1], 'Humerothoracic Elevation (Deg)', None)

    # set axes limits
    ax_limits_left = [(-75, 10), (-90, 15), (-150, 35)]
    ax_limits_right = [(-40, 62), (-30, 55), (-50, 40)]
    for i in range(3):
        axs_diff[i, 0].set_ylim(ax_limits_left[i][0], ax_limits_left[i][1])
        axs_diff[i, 1].set_ylim(ax_limits_right[i][0], ax_limits_right[i][1])
        axs_diff[i, 0].yaxis.set_major_locator(ticker.MultipleLocator(20))
        axs_diff[i, 1].yaxis.set_major_locator(ticker.MultipleLocator(20))

    # plot
    spm_y = np.array([[5, 55], [10, 50], [30, 35]])
    max_pos = 140
    leg_left_mean = []
    leg_right_mean = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_isb = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['ht', 'common_fine_up', 'euler.gh_isb', 2, 'common_fine_up']), axis=0)
        all_traj_isb_poe = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['ht', 'common_fine_up', 'euler.gh_isb', 0, 'common_fine_up']), axis=0)
        all_traj_phadke = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['ht', 'common_fine_up', 'euler.gh_phadke', 2, 'common_fine_up']), axis=0)
        all_traj_true = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['ht', 'common_fine_up', 'true_axial_rot', None, 'common_fine_up']), axis=0)
        all_traj_isb_norm = all_traj_isb + all_traj_isb_poe

        all_traj_isb_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['ht', 'euler.gh_isb', 2, 'common_fine_up']), axis=0)
        all_traj_isb_poe_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['ht', 'euler.gh_isb', 0, 'common_fine_up']), axis=0)
        all_traj_phadke_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['ht', 'euler.gh_phadke', 2, 'common_fine_up']), axis=0)
        all_traj_true_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['ht', 'true_axial_rot', None, 'common_fine_up']), axis=0)
        all_traj_isb_norm_max = all_traj_isb_max + all_traj_isb_poe_max

        # means and standard deviations
        isb_mean = np.rad2deg(np.mean(all_traj_isb, axis=0))
        isb_norm_mean = np.rad2deg(np.mean(all_traj_isb_norm, axis=0))
        phadke_mean = np.rad2deg(np.mean(all_traj_phadke, axis=0))
        true_mean = np.rad2deg(np.mean(all_traj_true, axis=0))
        isb_sd = np.rad2deg(np.std(all_traj_isb, ddof=1, axis=0))
        phadke_sd = np.rad2deg(np.std(all_traj_phadke, ddof=1, axis=0))
        true_sd = np.rad2deg(np.std(all_traj_true, ddof=1, axis=0))
        isb_norm_sd = np.rad2deg(np.std(all_traj_isb_norm, ddof=1, axis=0))

        isb_mean_max = np.rad2deg(np.mean(all_traj_isb_max, axis=0))
        isb_norm_mean_max = np.rad2deg(np.mean(all_traj_isb_norm_max, axis=0))
        phadke_mean_max = np.rad2deg(np.mean(all_traj_phadke_max, axis=0))
        true_mean_max = np.rad2deg(np.mean(all_traj_true_max, axis=0))
        isb_sd_max = np.rad2deg(np.std(all_traj_isb_max, ddof=1, axis=0))
        isb_norm_sd_max = np.rad2deg(np.std(all_traj_isb_norm_max, ddof=1, axis=0))
        phadke_sd_max = np.rad2deg(np.std(all_traj_phadke_max, ddof=1, axis=0))
        true_sd_max = np.rad2deg(np.std(all_traj_true_max, ddof=1, axis=0))

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        true_ln_left = mean_sd_plot(axs_diff[cur_row, 0], x, true_mean, true_sd,
                                    dict(color=color_map.colors[2], alpha=0.3, hatch='...'),
                                    dict(color=color_map.colors[2], marker=markers[2], markevery=20))
        phadke_ln = mean_sd_plot(axs_diff[cur_row, 0], x, phadke_mean, phadke_sd,
                                 dict(color=color_map.colors[1], alpha=0.25),
                                 dict(color=color_map.colors[1], marker=markers[1], markevery=20))

        true_ln_right = mean_sd_plot(axs_diff[cur_row, 1], x, true_mean, true_sd,
                                     dict(color=color_map.colors[2], alpha=0.3, hatch='...'),
                                     dict(color=color_map.colors[2], marker=markers[2], markevery=20))
        isb_ln = mean_sd_plot(axs_diff[cur_row, 1], x, isb_mean, isb_sd,
                              dict(color=color_map.colors[0], alpha=0.25),
                              dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        isb_norm_ln = mean_sd_plot(axs_diff[cur_row, 1], x, isb_norm_mean, isb_norm_sd,
                                   dict(color=color_map.colors[3], alpha=0.25),
                                   dict(color=color_map.colors[3], marker=markers[3], markevery=20))

        # plot endpoints
        axs_diff[cur_row, 0].errorbar(max_pos - 2, true_mean_max, yerr=true_sd_max,
                                      color=color_map.colors[2], marker=markers[2], capsize=3)
        axs_diff[cur_row, 0].errorbar(max_pos + 2, phadke_mean_max, yerr=phadke_sd_max,
                                      color=color_map.colors[1], marker=markers[1], capsize=3)

        axs_diff[cur_row, 1].errorbar(max_pos, true_mean_max, yerr=true_sd_max,
                                      color=color_map.colors[2], marker=markers[2], capsize=3)
        axs_diff[cur_row, 1].errorbar(max_pos - 3, isb_mean_max, yerr=isb_sd_max,
                                      color=color_map.colors[0], marker=markers[0], capsize=3)
        axs_diff[cur_row, 1].errorbar(max_pos + 3, isb_norm_mean_max, yerr=isb_norm_sd_max,
                                      color=color_map.colors[3], marker=markers[3], capsize=3)

        # spm
        phadke_vs_true = spm_test(all_traj_phadke[:, 1:], all_traj_true[:, 1:]).inference(alpha, two_tailed=True,
                                                                                          **infer_params)
        isb_vs_true = spm_test(all_traj_isb[:, 1:], all_traj_true[:, 1:]).inference(alpha, two_tailed=True,
                                                                                    **infer_params)
        isb_norm_vs_true = spm_test(all_traj_isb_norm[:, 1:], all_traj_true[:, 1:]).inference(alpha, two_tailed=True,
                                                                                              **infer_params)

        # plot spm
        phadke_x_sig = sig_filter(phadke_vs_true, x[1:])
        isb_x_sig = sig_filter(isb_vs_true, x[1:])
        isb_norm_x_sig = sig_filter(isb_norm_vs_true, x[1:])
        axs_diff[cur_row, 0].plot(phadke_x_sig, np.repeat(spm_y[cur_row, 0], phadke_x_sig.size), '-*',
                                  color=color_map.colors[1], markevery=20, lw=2)
        axs_diff[cur_row, 1].plot(isb_x_sig, np.repeat(spm_y[cur_row, 1], isb_x_sig.size), '-*',
                                  color=color_map.colors[0], markevery=20, lw=2)
        axs_diff[cur_row, 1].plot(isb_norm_x_sig, np.repeat(spm_y[cur_row, 1] - 5, isb_norm_x_sig.size), '-*',
                                  color=color_map.colors[3], markevery=20, lw=2)

        print('Activity: {}'.format(activity))
        print('ISB')
        print(extract_sig(isb_vs_true, x))
        print('ISB Norm')
        print(extract_sig(isb_norm_vs_true, x))
        print('Phadke')
        print(extract_sig(phadke_vs_true, x))

        if idx == 0:
            leg_left_mean.append(true_ln_left[0])
            leg_left_mean.append(phadke_ln[0])
            leg_right_mean.append(true_ln_right[0])
            leg_right_mean.append(isb_ln[0])
            leg_right_mean.append(isb_norm_ln[0])

    # figure title and legend
    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
    fig_diff.suptitle('Apparent vs True HT Axial Rotation for Arm Elevation', x=0.46, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig_diff.legend(leg_left_mean, ['True Axial', "xz'y''"], loc='upper left',
                               bbox_to_anchor=(0, 1), ncol=1, handlelength=1.5, handletextpad=0.5,
                               columnspacing=0.75, borderpad=0.2)
    leg_right = fig_diff.legend(leg_right_mean[:-1], ["True Axial", "yx'y''"],
                                loc='upper right', bbox_to_anchor=(1, 1), ncol=2, handlelength=1.5,
                                labelspacing=0.4, handletextpad=0.5, columnspacing=0.75, borderpad=0.2)

    # this is a hack so the yx'y'' PoE Adjusted label spans multiple columns
    leg_right2 = fig_diff.legend([leg_right_mean[-1]], ["yx'y'' PoE Adjusted"])
    leg_right2.remove()
    leg_right._legend_box._children.append(leg_right2._legend_handle_box)
    leg_right2._legend_box.stale = True

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
    x_ticks = np.concatenate((x_ticks, [max_pos]))
    for row in axs_diff:
        for ax in row:
            ax.set_xticks(x_ticks)
            tick_labels = [str(i) for i in x_ticks]
            tick_labels[-1] = 'Max'
            ax.set_xticklabels(tick_labels)

    # add axes titles
    _, y0, _, h = axs_diff[0, 0].get_position().bounds
    fig_diff.text(0.5, y0 + h * 1.03, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff[1, 0].get_position().bounds
    fig_diff.text(0.5, y0 + h * 1.03, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff[2, 0].get_position().bounds
    fig_diff.text(0.5, y0 + h * 1.03, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    # add arrows indicating direction
    axs_diff[0, 0].arrow(35, -27, 0, -25, length_includes_head=True, head_width=2, head_length=2)
    axs_diff[0, 0].text(23, -25, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    if params.fig_file:
        fig_diff.savefig(params.fig_file)
    else:
        plt.show()