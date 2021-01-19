"""Compare yx'y'' and zx'y'' GH axial rotation against true axial rotation.

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
    from true_vs_apparent.common.plot_utils import (init_graphing, mean_sd_plot, spm_plot_alpha, HandlerTupleVertical,
                                                    extract_sig, style_axes)
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db, extract_sub_rot_norm
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare yx'y'' and zx'y'' GH axial rotation against true axial rotation ",
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
    use_ac = bool(distutils.util.strtobool(params.use_ac))

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, use_ac, params.dtheta_fine, params.dtheta_coarse,
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

    fig_diff_norm = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs_diff_norm = fig_diff_norm.subplots(3, 2)

    # style axes, add x and y labels
    style_axes(axs_diff_norm[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff_norm[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_diff_norm[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_diff_norm[0, 1], None, 'SPM{t}')
    style_axes(axs_diff_norm[1, 1], None, 'SPM{t}')
    style_axes(axs_diff_norm[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{t}')

    # plot
    leg_patch_mean_norm = []
    leg_patch_t_norm = []
    alpha_patches = []
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
        isb_vs_true = spm_test(all_traj_isb, all_traj_true).inference(alpha, two_tailed=True, **infer_params)
        phadke_vs_true = spm_test(all_traj_phadke, all_traj_true).inference(alpha, two_tailed=True, **infer_params)

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        isb_ln = mean_sd_plot(axs_diff_norm[cur_row, 0], x, isb_mean, isb_sd,
                              dict(color=color_map.colors[0], alpha=0.25),
                              dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        true_ln = mean_sd_plot(axs_diff_norm[cur_row, 0], x, true_mean, true_sd,
                               dict(color=color_map.colors[2], alpha=0.3),
                               dict(color=color_map.colors[2], marker=markers[2], markevery=20))
        phadke_ln = mean_sd_plot(axs_diff_norm[cur_row, 0], x, phadke_mean, phadke_sd,
                                 dict(color=color_map.colors[1], alpha=0.25),
                                 dict(color=color_map.colors[1], marker=markers[1], markevery=20))

        # plot spm
        isb_t_ln, isb_alpha = spm_plot_alpha(axs_diff_norm[cur_row, 1], x, isb_vs_true,
                                             dict(color=color_map.colors[0], alpha=0.25),
                                             dict(color=color_map.colors[0]))
        phadke_t_ln, phadke_alpha = spm_plot_alpha(axs_diff_norm[cur_row, 1], x, phadke_vs_true,
                                                   dict(color=color_map.colors[1], alpha=0.25),
                                                   dict(color=color_map.colors[1]))

        print('Activity: {}'.format(activity))
        print('ISB')
        print(extract_sig(isb_vs_true, x))
        print('Phadke')
        print(extract_sig(phadke_vs_true, x))

        if idx == 0:
            leg_patch_mean_norm.append(isb_ln[0])
            leg_patch_mean_norm.append(true_ln[0])
            leg_patch_mean_norm.append(phadke_ln[0])
            leg_patch_t_norm.append(isb_t_ln[0])
            leg_patch_t_norm.append(phadke_t_ln[0])
            alpha_patches.append((isb_alpha, phadke_alpha))

    # figure title and legend
    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
    fig_diff_norm.suptitle('Apparent vs True GH Axial Rotation Comparison', x=0.492, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig_diff_norm.legend(leg_patch_mean_norm, ["yx'y'' (ISB)", 'True Axial', "xz'y''"], loc='upper left',
                                    bbox_to_anchor=(0, 1), ncol=2, handlelength=1.5, handletextpad=0.5,
                                    columnspacing=0.75, borderpad=0.2)
    leg_right = fig_diff_norm.legend(leg_patch_t_norm + alpha_patches,
                                     ["yx'y'' (ISB) = True Axial", "xz'y'' = True Axial", '$\\alpha=0.05$'],
                                     handler_map={tuple: HandlerTupleVertical(ndivide=None)},
                                     loc='upper right', bbox_to_anchor=(1, 1), ncol=1, handlelength=1.5,
                                     labelspacing=0.4, handletextpad=0.5, columnspacing=0.75, borderpad=0.2)

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
    for row in axs_diff_norm:
        for ax in row:
            ax.set_xticks(x_ticks)

    # add axes titles
    _, y0, _, h = axs_diff_norm[0, 0].get_position().bounds
    fig_diff_norm.text(0.5, y0 + h * 1.03, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff_norm[1, 0].get_position().bounds
    fig_diff_norm.text(0.5, y0 + h * 1.03, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_diff_norm[2, 0].get_position().bounds
    fig_diff_norm.text(0.5, y0 + h * 1.03, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    # add arrows indicating direction
    axs_diff_norm[0, 0].arrow(35, 45, 0, -20, length_includes_head=True, head_width=2, head_length=2)
    axs_diff_norm[0, 0].text(23, 45, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    if params.fig_file:
        fig_diff_norm.savefig(params.fig_file)
    else:
        plt.show()
