"""Compare GH yx'y'', xz'y'' and true axial rotation between planes of elevation.

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
    import spm1d
    from true_vs_apparent.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, extract_sig,
                                                    style_axes, sig_filter, output_spm_p)
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import (prepare_db, extract_sub_rot_norm, sub_rot_at_max_elev,
                                                        subj_name_to_number)
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare GH yx'y'', xz'y'' and true axial rotation between planes of elevation",
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

    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, 'GC', params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_elev_equal = db_elev.loc[~db_elev['Trial_Name'].str.contains('U35_010')].copy()

    #%%
    if bool(distutils.util.strtobool(params.parametric)):
        post_hoc_spm_test = spm1d.stats.ttest_paired
        spm_test = spm1d.stats.ttest
        infer_params = {}
    else:
        post_hoc_spm_test = spm1d.stats.nonparam.ttest_paired
        spm_test = spm1d.stats.nonparam.ttest
        infer_params = {'force_iterations': True}

    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_order = {'CA': -1, 'SA': 0, 'FE': 1}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(90 / 25.4, 230 / 25.4), dpi=params.dpi)
    axs = fig.subplots(4, 1)

    # style axes, add x and y labels
    style_axes(axs[0], None, 'Axial Rotation (Deg)')
    style_axes(axs[1], None, 'Axial Rotation (Deg)')
    style_axes(axs[2], None, 'Axial Rotation (Deg)')
    style_axes(axs[3], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')

    # set axes limits
    for i in range(3):
        axs[i].set_ylim(-45, 55)

    all_traj_isb_rm = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
    all_traj_isb_poe_rm = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 0, 'up']), axis=0)
    all_traj_phadke_rm = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
    all_traj_true_rm = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)
    group_rm = (db_elev_equal['Activity'].map({'CA': 1, 'SA': 2, 'FE': 3})).to_numpy(dtype=np.int)
    all_traj_isb_norm_rm = all_traj_isb_rm + all_traj_isb_poe_rm

    subj_rm = (db_elev_equal['Subject_Short'].map(subj_name_to_number)).to_numpy()
    spm_one_way_rm_true = spm1d.stats.anova1rm(all_traj_true_rm, group_rm, subj_rm).inference(alpha=0.05)
    spm_one_way_rm_isb = spm1d.stats.anova1rm(all_traj_isb_rm, group_rm, subj_rm).inference(alpha=0.05)
    spm_one_way_rm_isb_norm = spm1d.stats.anova1rm(all_traj_isb_norm_rm, group_rm, subj_rm).inference(alpha=0.05)
    spm_one_way_rm_phadke = spm1d.stats.anova1rm(all_traj_phadke_rm, group_rm, subj_rm).inference(alpha=0.05)

    true_x_sig = sig_filter(spm_one_way_rm_true, x)
    isb_x_sig = sig_filter(spm_one_way_rm_isb, x)
    isb_norm_x_sig = sig_filter(spm_one_way_rm_isb_norm, x)
    phadke_x_sig = sig_filter(spm_one_way_rm_phadke, x)

    axs[0].plot(true_x_sig, np.repeat(50, true_x_sig.size), color='k', lw=2)
    axs[1].plot(phadke_x_sig, np.repeat(50, phadke_x_sig.size), color='k', lw=2)
    axs[2].plot(isb_x_sig, np.repeat(50, isb_x_sig.size), color='k', lw=2)
    axs[3].plot(isb_norm_x_sig, np.repeat(50, isb_norm_x_sig.size), color='k', lw=2)

    print('ANOVA')
    print('True Axial')
    print(extract_sig(spm_one_way_rm_true, x))
    print('P-values: ')
    print(output_spm_p(spm_one_way_rm_true))
    print('ISB')
    print(extract_sig(spm_one_way_rm_isb, x))
    print('P-values: ')
    print(output_spm_p(spm_one_way_rm_isb))
    print('ISB Norm')
    print(extract_sig(spm_one_way_rm_isb_norm, x))
    print('P-values: ')
    print(output_spm_p(spm_one_way_rm_isb_norm))
    print('Phadke')
    print(extract_sig(spm_one_way_rm_phadke, x))
    print('P-values: ')
    print(output_spm_p(spm_one_way_rm_phadke))

    print('ANOVA Post-hoc')
    p_critical = spm1d.util.p_critical_bonf(alpha, 3)
    db_ca = db_elev_equal.loc[db['Trial_Name'].str.contains('_CA_')].copy()
    db_sa = db_elev_equal.loc[db['Trial_Name'].str.contains('_SA_')].copy()
    db_fe = db_elev_equal.loc[db['Trial_Name'].str.contains('_FE_')].copy()

    def extract_isb(db):
        return np.stack(db['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)

    def extract_phadke(db):
        return np.stack(db['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)

    def extract_isb_norm(db):
        all_traj_isb = np.stack(db['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_isb_poe = np.stack(db['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 0, 'up']), axis=0)
        return all_traj_isb + all_traj_isb_poe

    decomp_map = {'ISB': extract_isb, 'Phadke': extract_phadke, 'ISB_Norm': extract_isb_norm}
    for decomp_name, decomp_fnc in decomp_map.items():
        print(decomp_name)
        ca = decomp_fnc(db_ca)
        sa = decomp_fnc(db_sa)
        fe = decomp_fnc(db_fe)

        ca_vs_sa = post_hoc_spm_test(ca, sa).inference(p_critical, two_tailed=True, **infer_params)
        ca_vs_fe = post_hoc_spm_test(ca, fe).inference(p_critical, two_tailed=True, **infer_params)
        sa_vs_fe = post_hoc_spm_test(sa, fe).inference(p_critical, two_tailed=True, **infer_params)

        print('CA vs SA')
        print(extract_sig(ca_vs_sa, x))
        print('P-values: ')
        print(output_spm_p(ca_vs_sa))
        print('CA vs FE')
        print(extract_sig(ca_vs_fe, x))
        print('P-values: ')
        print(output_spm_p(ca_vs_fe))
        print('SA vs FE')
        print(extract_sig(sa_vs_fe, x))
        print('P-values: ')
        print(output_spm_p(sa_vs_fe))

    # plot
    max_pos = 140
    mean_lns = []
    t_lns = []
    activities = []
    act_order_t = {'ca': 1, 'sa': 2, 'fe': 3}
    print('AGAINST ZERO')
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        print(activity)
        all_traj_isb_act = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_isb_poe_act = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 0, 'up']), axis=0)
        all_traj_phadke_act = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true_act = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)
        all_traj_isb_act_norm = all_traj_isb_act + all_traj_isb_poe_act

        all_traj_isb_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'euler.gh_isb', 2, 'up']), axis=0)
        all_traj_isb_poe_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'euler.gh_isb', 0, 'up']), axis=0)
        all_traj_phadke_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'true_axial_rot', None, 'up']), axis=0)
        all_traj_isb_norm_max = all_traj_isb_max + all_traj_isb_poe_max

        isb_mean_act = np.rad2deg(np.mean(all_traj_isb_act, axis=0))
        isb_mean_norm_act = np.rad2deg(np.mean(all_traj_isb_act_norm, axis=0))
        phadke_mean_act = np.rad2deg(np.mean(all_traj_phadke_act, axis=0))
        true_mean_act = np.rad2deg(np.mean(all_traj_true_act, axis=0))
        isb_sd_act = np.rad2deg(np.std(all_traj_isb_act, ddof=1, axis=0))
        isb_sd_norm_act = np.rad2deg(np.std(all_traj_isb_act_norm, ddof=1, axis=0))
        phadke_sd_act = np.rad2deg(np.std(all_traj_phadke_act, ddof=1, axis=0))
        true_sd_act = np.rad2deg(np.std(all_traj_true_act, ddof=1, axis=0))

        isb_mean_max = np.rad2deg(np.mean(all_traj_isb_max, axis=0))
        isb_norm_mean_max = np.rad2deg(np.mean(all_traj_isb_norm_max, axis=0))
        phadke_mean_max = np.rad2deg(np.mean(all_traj_phadke_max, axis=0))
        true_mean_max = np.rad2deg(np.mean(all_traj_true_max, axis=0))
        isb_sd_max = np.rad2deg(np.std(all_traj_isb_max, ddof=1, axis=0))
        isb_norm_sd_max = np.rad2deg(np.std(all_traj_isb_norm_max, ddof=1, axis=0))
        phadke_sd_max = np.rad2deg(np.std(all_traj_phadke_max, ddof=1, axis=0))
        true_sd_max = np.rad2deg(np.std(all_traj_true_max, ddof=1, axis=0))

        # plot mean +- sd
        shaded = dict(color=color_map.colors[idx], alpha=0.25)
        line = dict(color=color_map.colors[idx], marker=markers[idx], markevery=20)
        if activity == 'SA':
            shaded['hatch'] = '...'
        true_ln = mean_sd_plot(axs[0], x, true_mean_act, true_sd_act, shaded, line)
        phadke_ln = mean_sd_plot(axs[1], x, phadke_mean_act, phadke_sd_act, shaded, line)
        isb_ln = mean_sd_plot(axs[2], x, isb_mean_act, isb_sd_act, shaded, line)
        isb_norm_ln = mean_sd_plot(axs[3], x, isb_mean_norm_act, isb_sd_norm_act, shaded, line)

        mean_lns.append(true_ln[0])
        activities.append(activity)

        # plot endpoints
        axs[0].errorbar(max_pos + act_order[activity] * 3, true_mean_max, yerr=true_sd_max,
                        color=color_map.colors[idx], marker=markers[2], capsize=3)
        axs[1].errorbar(max_pos + act_order[activity] * 3, phadke_mean_max, yerr=phadke_sd_max,
                        color=color_map.colors[idx], marker=markers[2], capsize=3)
        axs[2].errorbar(max_pos + act_order[activity] * 3, isb_mean_max, yerr=isb_sd_max,
                        color=color_map.colors[idx], marker=markers[2], capsize=3)
        axs[3].errorbar(max_pos + act_order[activity] * 3, isb_norm_mean_max, yerr=isb_norm_sd_max,
                        color=color_map.colors[idx], marker=markers[2], capsize=3)

        # spm
        true_vs_zero = spm_test(all_traj_true_act, 0).inference(alpha, two_tailed=True, **infer_params)
        phadke_vs_zero = spm_test(all_traj_phadke_act, 0).inference(alpha, two_tailed=True, **infer_params)
        isb_vs_zero = spm_test(all_traj_isb_act, 0).inference(alpha, two_tailed=True, **infer_params)
        isb_norm_vs_zero = spm_test(all_traj_isb_act_norm, 0).inference(alpha, two_tailed=True, **infer_params)

        print('At Max')
        print('Mean {:.2f} SD: {:.2f}'.format(true_mean_max, true_sd_max))
        print('True Axial')
        print(extract_sig(true_vs_zero, x))
        print('P-values: ')
        print(output_spm_p(true_vs_zero))
        print('ISB')
        print(extract_sig(isb_vs_zero, x))
        print('P-values: ')
        print(output_spm_p(isb_vs_zero))
        print('ISB Norm')
        print(extract_sig(isb_norm_vs_zero, x))
        print('P-values: ')
        print(output_spm_p(isb_norm_vs_zero))
        print('Phadke')
        print(extract_sig(phadke_vs_zero, x))
        print('P-values: ')
        print(output_spm_p(phadke_vs_zero))

        # plot spm
        true_zero_x_sig = sig_filter(true_vs_zero, x)
        phadke_zero_x_sig = sig_filter(phadke_vs_zero, x)
        isb_zero_x_sig = sig_filter(isb_vs_zero, x)
        isb_norm_zero_x_sig = sig_filter(isb_norm_vs_zero, x)
        true_zro_t_ln = axs[0].plot(true_zero_x_sig, np.repeat(50 - act_order_t[activity.lower()] * 3,
                                                               true_zero_x_sig.size), color=color_map.colors[idx], lw=2)
        axs[1].plot(phadke_zero_x_sig, np.repeat(50 - act_order_t[activity.lower()] * 3, phadke_zero_x_sig.size),
                    color=color_map.colors[idx], lw=2)
        axs[2].plot(isb_zero_x_sig, np.repeat(50 - act_order_t[activity.lower()] * 3, isb_zero_x_sig.size),
                    color=color_map.colors[idx], lw=2)
        axs[3].plot(isb_norm_zero_x_sig, np.repeat(50 - act_order_t[activity.lower()] * 3, isb_norm_zero_x_sig.size),
                    color=color_map.colors[idx], lw=2)

        t_lns.append(true_zro_t_ln[0])

    # figure title and legend
    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
    fig.suptitle('GH Axial Rotation\nComparison by Plane', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig.legend([mean_lns[i] for i in [0, 2, 1]], [activities[i] for i in [0, 2, 1]], loc='upper right',
                          bbox_to_anchor=(1, 0.97), ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                          borderpad=0.2, labelspacing=0.4)

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
    for ax in axs:
        ax.set_xticks(x_ticks)
        tick_labels = [str(i) for i in x_ticks]
        tick_labels[-1] = 'Max'
        ax.set_xticklabels(tick_labels)

    # add arrows indicating direction
    axs[0].arrow(37, 38, 0, -30, length_includes_head=True, head_width=2, head_length=2)
    axs[0].text(23, 40, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs[0].get_position().bounds
    fig.text(0.5, y0 + h * 0.98, 'True Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, "xz'y'' Axial Rotation", ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[2].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, "yx'y'' (ISB) Axial Rotation", ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[3].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, "PoE Adjusted yx'y'' Axial Rotation", ha='center', fontsize=11, fontweight='bold')
    make_interactive()

    # normality tests
    spm_one_way_rm_true_normality = spm1d.stats.normality.sw.anova1rm(all_traj_true_rm, group_rm, subj_rm)
    spm_one_way_rm_isb_normality = spm1d.stats.normality.sw.anova1rm(all_traj_isb_rm, group_rm, subj_rm)
    spm_one_way_rm_isb_norm_normality = spm1d.stats.normality.sw.anova1rm(all_traj_isb_norm_rm, group_rm, subj_rm)
    spm_one_way_rm_phadke_normality = spm1d.stats.normality.sw.anova1rm(all_traj_phadke_rm, group_rm, subj_rm)

    fig_norm = plt.figure()
    ax_norm = fig_norm.subplots()
    ax_norm.axhline(0.05, ls='--', color='grey')
    norm_rm_true_ln = ax_norm.plot(x, spm_one_way_rm_true_normality[1], color=color_map.colors[0], ls='--')
    norm_rm_isb_ln = ax_norm.plot(x, spm_one_way_rm_isb_normality[1], color=color_map.colors[1], ls='--')
    norm_rm_isb_norm_ln = ax_norm.plot(x, spm_one_way_rm_isb_norm_normality[1], color=color_map.colors[3], ls='--')
    norm_rm_phadke_ln = ax_norm.plot(x, spm_one_way_rm_phadke_normality[1], color=color_map.colors[2],
                                     ls='--')
    fig_norm.legend([norm_rm_true_ln[0], norm_rm_isb_ln[0], norm_rm_isb_norm_ln[0], norm_rm_phadke_ln[0]],
                    ['True', 'ISB', 'ISB Norm', 'Phadke'], loc='upper right', ncol=2,
                    handlelength=1.5, handletextpad=0.5, columnspacing=0.75)
    style_axes(ax_norm, 'Humerothoracic Elevation (Deg)', 'p-value')
    plt.tight_layout()
    fig_norm.suptitle('Normality tests')

    print('Normality True: {:.2f}'.format(x[np.nonzero(spm_one_way_rm_true_normality[1] > 0.05)[0][0]]))
    print('Normality ISB: {:.2f}'.format(x[np.nonzero(spm_one_way_rm_isb_normality[1] > 0.05)[0][0]]))
    print('Normality ISB Norm: {:.2f}'.format(x[np.nonzero(spm_one_way_rm_isb_norm_normality[1] > 0.05)[0][0]]))
    print('Normality Phadke: {:.2f}'.format(x[np.nonzero(spm_one_way_rm_phadke_normality[1] > 0.05)[0][0]]))

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
