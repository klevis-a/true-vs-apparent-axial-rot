"""Compare GH yx'y'', yx'y'' normalized, and xz'y'' against true axial rotation for external rotation trials.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
torso_def: Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition.
dtheta_fine: Incremental angle (deg) to use for fine interpolation between minimum and maximum HT elevation analyzed.
dtheta_coarse: Incremental angle (deg) to use for coarse interpolation between minimum and maximum HT elevation analyzed.
backend: Matplotlib backend to use for plotting (e.g. Qt5Agg, macosx, etc.).
era90_endpts: Path to csv file containing start and stop frames (including both external and internal rotation) for
external rotation in 90 deg of abduction trials.
erar_endpts: Path to csv file containing start and stop frames (including both external and internal rotation) for
external rotation in adduction trials.
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
    import spm1d
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    from true_vs_apparent.common import plot_utils
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    from true_vs_apparent.common.analysis_er_utils import ready_er_db
    from true_vs_apparent.common.plot_utils import (mean_sd_plot, make_interactive, style_axes, extract_sig, sig_filter,
                                                    output_spm_p)
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare GH yx'y'', yx'y'' normalized, and xz'y'' against true axial rotation for "
                                     "external rotation trials", __package__, __file__))
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

    # ready db
    db_er_endpts = ready_er_db(db, params.torso_def, 'GC', params.erar_endpts, params.era90_endpts,
                               params.dtheta_fine)

#%%
    if bool(distutils.util.strtobool(params.parametric)):
        spm_test = spm1d.stats.ttest_paired
        infer_params = {}
    else:
        spm_test = spm1d.stats.nonparam.ttest_paired
        infer_params = {'force_iterations': True}

    x = np.arange(0, 100 + params.dtheta_fine, params.dtheta_fine)
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    plot_utils.init_graphing(params.backend)
    plt.close('all')
    fig = plt.figure(figsize=(190 / 25.4, 150 / 25.4), dpi=params.dpi)
    axs = fig.subplots(2, 2)

    ax_limits = [(-140, 15), (-110, 30)]
    for row_idx, row in enumerate(axs):
        for col_idx, ax in enumerate(row):
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
            ax.yaxis.set_major_locator(plticker.MultipleLocator(base=25.0))
            x_label = 'Percent Complete (%)' if row_idx == 1 else None
            y_label = 'Axial Rotation (Deg)' if col_idx == 0 else None
            style_axes(ax, x_label, y_label)
            axs[row_idx, col_idx].set_ylim(ax_limits[row_idx][0], ax_limits[row_idx][1])

    spm_y = np.array([[13, 13], [27, 27]])
    traj_name = 'gh'
    mean_left_lns = []
    mean_right_lns = []
    for idx_act, (activity, activity_df) in enumerate(db_er_endpts.groupby('Activity', observed=True)):
        all_traj_true = np.stack(activity_df[traj_name + '_true'], axis=0)
        all_traj_isb = np.stack(activity_df[traj_name + '_isb'], axis=0)
        all_traj_isb_norm = np.stack(activity_df[traj_name + '_isb_norm'], axis=0)
        all_traj_phadke = np.stack(activity_df[traj_name + '_phadke'], axis=0)

        # means
        true_mean = np.rad2deg(np.mean(all_traj_true, axis=0))
        isb_mean = np.rad2deg(np.mean(all_traj_isb, axis=0))
        isb_norm_mean = np.rad2deg(np.mean(all_traj_isb_norm, axis=0))
        phadke_mean = np.rad2deg(np.mean(all_traj_phadke, axis=0))

        # sds
        true_sd = np.rad2deg(np.std(all_traj_true, ddof=1, axis=0))
        isb_sd = np.rad2deg(np.std(all_traj_isb, ddof=1, axis=0))
        isb_norm_sd = np.rad2deg(np.std(all_traj_isb_norm, ddof=1, axis=0))
        phadke_sd = np.rad2deg(np.std(all_traj_phadke, ddof=1, axis=0))

        # plots mean +- sd
        phadke_ln = mean_sd_plot(axs[idx_act, 0], x, phadke_mean, phadke_sd,
                                 dict(color=color_map.colors[1], alpha=0.2),
                                 dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        true_left_ln = mean_sd_plot(axs[idx_act, 0], x, true_mean, true_sd,
                                    dict(color=color_map.colors[2], alpha=0.2, hatch='...'),
                                    dict(color=color_map.colors[2], marker=markers[2], markevery=20))
        isb_ln = mean_sd_plot(axs[idx_act, 1], x, isb_mean, isb_sd,
                              dict(color=color_map.colors[0], alpha=0.2),
                              dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        isb_norm_ln = mean_sd_plot(axs[idx_act, 1], x, isb_norm_mean, isb_norm_sd,
                                   dict(color=color_map.colors[3], alpha=0.2),
                                   dict(color=color_map.colors[3], marker=markers[2], markevery=20))
        true_right_ln = mean_sd_plot(axs[idx_act, 1], x, true_mean, true_sd,
                                     dict(color=color_map.colors[2], alpha=0.2, hatch='...'),
                                     dict(color=color_map.colors[2], marker=markers[2], markevery=20))

        # spm
        isb_vs_true = spm_test(all_traj_isb[:, 1:],
                               all_traj_true[:, 1:]).inference(alpha, two_tailed=True, **infer_params)
        isb_norm_vs_true = spm_test(all_traj_isb_norm[:, 1:],
                                    all_traj_true[:, 1:]).inference(alpha, two_tailed=True, **infer_params)
        phadke_vs_true = spm_test(all_traj_phadke[:, 1:],
                                  all_traj_true[:, 1:]).inference(alpha, two_tailed=True, **infer_params)

        # plot spm
        phadke_x_sig = sig_filter(phadke_vs_true, x[1:])
        isb_x_sig = sig_filter(isb_vs_true, x[1:])
        isb_norm_x_sig = sig_filter(isb_norm_vs_true, x[1:])
        axs[idx_act, 0].plot(phadke_x_sig, np.repeat(spm_y[idx_act, 0], phadke_x_sig.size),
                             color=color_map.colors[1], lw=2)
        axs[idx_act, 1].plot(isb_x_sig, np.repeat(spm_y[idx_act, 1], isb_x_sig.size),
                             color=color_map.colors[0], lw=2)
        axs[idx_act, 1].plot(isb_norm_x_sig, np.repeat(spm_y[idx_act, 1] - 3, isb_norm_x_sig.size),
                             color=color_map.colors[3], lw=2)

        # print significance
        print('Activity: {} Joint: {}'.format(activity, traj_name.upper()))
        print('ISB vs True')
        print(extract_sig(isb_vs_true, x))
        print('Max: {:.2f}'.format(np.abs(isb_mean[-1] - true_mean[-1])))
        print('P-values: ')
        print(output_spm_p(isb_vs_true))
        print('ISB Rectified vs True')
        print(extract_sig(isb_norm_vs_true, x))
        print('Max: {:.2f}'.format(np.abs(isb_norm_mean[-1] - true_mean[-1])))
        print('P-values: ')
        print(output_spm_p(isb_norm_vs_true))
        print('Phadke vs True')
        print(extract_sig(phadke_vs_true, x))
        print('Max: {:.2f}'.format(np.abs(phadke_mean[-1] - true_mean[-1])))
        print('P-values: ')
        print(output_spm_p(phadke_vs_true))

        if idx_act == 0:
            # legend lines
            mean_left_lns.append(true_left_ln[0])
            mean_left_lns.append(phadke_ln[0])
            mean_right_lns.append(true_right_ln[0])
            mean_right_lns.append(isb_ln[0])
            mean_right_lns.append(isb_norm_ln[0])

    # figure title and legend
    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=-0.5)
    fig.suptitle('Apparent vs True GH Axial Rotation for ER-ADD and ER-ABD', x=0.4515, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.925)
    leg_left = fig.legend(mean_left_lns, ["True Axial", "xz'y''"], loc='upper left',
                          bbox_to_anchor=(0, 1), ncol=1, handlelength=1.2, handletextpad=0.3, columnspacing=0.6,
                          borderpad=0.2, labelspacing=0.4)
    leg_right = fig.legend(mean_right_lns, ['True Axial', "yx'y''"], loc='upper right',
                           bbox_to_anchor=(1, 1), ncol=2, handlelength=1.2, handletextpad=0.3, columnspacing=0.6,
                           labelspacing=0.3, borderpad=0.2)

    # this is a hack so the yx'y'' PoE Adjusted label spans multiple columns
    leg_right2 = fig.legend([mean_right_lns[-1]], ["Swing-Spin"])
    leg_right2.remove()
    leg_right._legend_box._children.append(leg_right2._legend_handle_box)
    leg_right2._legend_box.stale = True

    # remove y labels on 2nd column
    for i in range(2):
        labels = [item.get_text() for item in axs[i, 1].get_xticklabels()]
        empty_string_labels = [''] * len(labels)
        axs[i, 1].set_yticklabels(empty_string_labels)

    # add arrows indicating direction
    axs[0, 0].arrow(10, -72, 0, -32, length_includes_head=True, head_width=2, head_length=2)
    axs[0, 0].text(0, -70, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs[0, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'ER-ADD GH Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'ER-ABD GH Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
