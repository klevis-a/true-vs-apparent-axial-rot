"""Compare yx'y'' and zx'y'' GH axial rotation between planes of elevation.

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
    from true_vs_apparent.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, spm_plot_alpha,
                                                    extract_sig, style_axes)
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db, extract_sub_rot_norm
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare yx'y'' and zx'y'' GH axial rotation between planes of elevation",
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
    db_elev_equal = db_elev.loc[~db_elev['Trial_Name'].str.contains('N020')].copy()

    #%%
    def subj_name_to_number(subject_name):
        if subject_name[-1] == 'A':
            return int(subject_name[-2])
        else:
            if subject_name[-2] == '1' or subject_name[-2] == '2':
                return int(subject_name[-2:])
            else:
                return int(subject_name[-1])

    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig_plane_start = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs_plane_start = fig_plane_start.subplots(3, 2)

    # style axes, add x and y labels
    style_axes(axs_plane_start[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_plane_start[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_plane_start[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_plane_start[0, 1], None, 'SPM{F}')
    style_axes(axs_plane_start[1, 1], None, 'SPM{F}')
    style_axes(axs_plane_start[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{F}')

    # set axes limits
    for i in range(3):
        axs_plane_start[i, 0].set_ylim(-45, 45)

    all_traj_isb_rm_start = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
    all_traj_phadke_rm_start = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
    all_traj_true_rm_start = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)
    group_rm = (db_elev_equal['Activity'].map({'CA': 1, 'SA': 2, 'FE': 3})).to_numpy(dtype=np.int)
    subj_rm = (db_elev_equal['Subject_Short'].map(subj_name_to_number)).to_numpy()
    spm_one_way_rm_true_start = spm1d.stats.anova1rm(all_traj_true_rm_start, group_rm, subj_rm).inference(alpha=0.05)
    spm_one_way_rm_isb_start = spm1d.stats.anova1rm(all_traj_isb_rm_start, group_rm, subj_rm).inference(alpha=0.05)
    spm_one_way_rm_phadke_start = spm1d.stats.anova1rm(all_traj_phadke_rm_start,
                                                       group_rm, subj_rm).inference(alpha=0.05)

    shaded_spm_rm = dict(color=color_map.colors[6], alpha=0.25)
    line_spm_rm = dict(color=color_map.colors[7])
    one_way_rm_ln_true_start, alpha_ln_true = spm_plot_alpha(axs_plane_start[0, 1], x, spm_one_way_rm_true_start,
                                                             shaded_spm_rm, line_spm_rm)
    one_way_rm_ln_isb_start, alpha_ln_isb = spm_plot_alpha(axs_plane_start[1, 1], x, spm_one_way_rm_isb_start,
                                                           shaded_spm_rm, line_spm_rm)
    one_way_rm_ln_phadke_start, alpha_ln_phadke = spm_plot_alpha(axs_plane_start[2, 1], x, spm_one_way_rm_phadke_start,
                                                                 shaded_spm_rm, line_spm_rm)

    print('True Axial')
    print(extract_sig(spm_one_way_rm_true_start, x))
    print('ISB')
    print(extract_sig(spm_one_way_rm_isb_start, x))
    print('Phadke')
    print(extract_sig(spm_one_way_rm_phadke_start, x))

    mean_lns_start = []
    activities_start = []
    for idx, (activity, activity_df) in enumerate(db_elev_equal.groupby('Activity', observed=True)):
        all_traj_isb_act = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_phadke_act = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true_act = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)

        isb_mean_act = np.rad2deg(np.mean(all_traj_isb_act, axis=0))
        phadke_mean_act = np.rad2deg(np.mean(all_traj_phadke_act, axis=0))
        true_mean_act = np.rad2deg(np.mean(all_traj_true_act, axis=0))
        isb_sd_act = np.rad2deg(np.std(all_traj_isb_act, ddof=1, axis=0))
        phadke_sd_act = np.rad2deg(np.std(all_traj_phadke_act, ddof=1, axis=0))
        true_sd_act = np.rad2deg(np.std(all_traj_true_act, ddof=1, axis=0))

        # plot mean +- sd
        shaded = dict(color=color_map.colors[idx], alpha=0.25)
        line = dict(color=color_map.colors[idx], marker=markers[idx], markevery=20)
        true_ln = mean_sd_plot(axs_plane_start[0, 0], x, true_mean_act, true_sd_act, shaded, line)
        isb_ln = mean_sd_plot(axs_plane_start[1, 0], x, isb_mean_act, isb_sd_act, shaded, line)
        phadke_ln = mean_sd_plot(axs_plane_start[2, 0], x, phadke_mean_act, phadke_sd_act, shaded, line)

        mean_lns_start.append(true_ln[0])
        activities_start.append(activity)

    # figure title and legend
    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
    fig_plane_start.suptitle('Glenohumeral Axial Rotation Comparison by Plane', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig_plane_start.legend(mean_lns_start, activities_start, loc='upper left', bbox_to_anchor=(0, 1),
                                      ncol=2, handlelength=1.5, handletextpad=0.5, columnspacing=0.75, borderpad=0.2,
                                      labelspacing=0.4)
    leg_right = fig_plane_start.legend([one_way_rm_ln_true_start[0], alpha_ln_true],
                                       ['CA = SA = FE', '$\\alpha=0.05$'], loc='upper right', borderpad=0.2,
                                       bbox_to_anchor=(1, 1), handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                                       labelspacing=0.4)

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
    for row in axs_plane_start:
        for ax in row:
            ax.set_xticks(x_ticks)

    # add arrows indicating direction
    axs_plane_start[0, 0].arrow(35, 40, 0, -30, length_includes_head=True, head_width=2, head_length=2)
    axs_plane_start[0, 0].text(23, 40, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs_plane_start[0, 0].get_position().bounds
    fig_plane_start.text(0.5, y0 + h * 1.02, 'True Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_plane_start[1, 0].get_position().bounds
    fig_plane_start.text(0.5, y0 + h * 1.02, "yx'y'' (ISB) Axial Rotation", ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_plane_start[2, 0].get_position().bounds
    fig_plane_start.text(0.5, y0 + h * 1.02, "xz'y'' Axial Rotation", ha='center', fontsize=11, fontweight='bold')
    make_interactive()

    # normality tests
    spm_one_way_rm_true_norm_start = spm1d.stats.normality.sw.anova1rm(all_traj_true_rm_start, group_rm, subj_rm)
    spm_one_way_rm_isb_norm_start = spm1d.stats.normality.sw.anova1rm(all_traj_isb_rm_start, group_rm, subj_rm)
    spm_one_way_rm_phadke_norm_start = spm1d.stats.normality.sw.anova1rm(all_traj_phadke_rm_start, group_rm, subj_rm)

    fig_norm_start = plt.figure()
    ax_norm_start = fig_norm_start.subplots()
    ax_norm_start.axhline(0.05, ls='--', color='grey')
    norm_rm_true_ln_start = ax_norm_start.plot(x, spm_one_way_rm_true_norm_start[1], color=color_map.colors[0], ls='--')
    norm_rm_isb_ln_start = ax_norm_start.plot(x, spm_one_way_rm_isb_norm_start[1], color=color_map.colors[1], ls='--')
    norm_rm_phadke_ln_start = ax_norm_start.plot(x, spm_one_way_rm_phadke_norm_start[1], color=color_map.colors[2],
                                                 ls='--')
    fig_norm_start.legend([norm_rm_true_ln_start[0], norm_rm_isb_ln_start[0], norm_rm_phadke_ln_start[0]],
                          ['True', 'ISB', 'Phadke'], loc='upper right', ncol=2,
                          handlelength=1.5, handletextpad=0.5, columnspacing=0.75)
    style_axes(ax_norm_start, 'Humerothoracic Elevation (Deg)', 'p-value')
    plt.tight_layout()
    fig_norm_start.suptitle('Normality tests')

    print('Normality True Axial Rotation: {:.2f}'.format(x[np.nonzero(spm_one_way_rm_true_norm_start[1] > 0.05)[0][0]]))
    print('Normality ISB Axial Rotation: {:.2f}'.format(x[np.nonzero(spm_one_way_rm_isb_norm_start[1] > 0.05)[0][0]]))
    print('Normality Phadke Axial Rotation: {:.2f}'.
          format(x[np.nonzero(spm_one_way_rm_phadke_norm_start[1] > 0.05)[0][0]]))

    if params.fig_file:
        fig_plane_start.savefig(params.fig_file)
    else:
        plt.show()
