"""Compare yx'y'', yx'y'' normalized, and xz'y'' against true axial rotation for external rotation trials.

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
    import spm1d
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    from true_vs_apparent.common import plot_utils
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    from true_vs_apparent.common.analysis_er_utils import ready_er_db
    from true_vs_apparent.common.plot_utils import (mean_sd_plot, make_interactive, style_axes, spm_plot_alpha,
                                                    HandlerTupleVertical, extract_sig)
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare yx'y'', yx'y'' normalized, and xz'y'' against true axial rotation for "
                                     "external rotation trials", __package__, __file__))
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

    # ready db
    db_er_endpts = ready_er_db(db, params.torso_def, use_ac, params.erar_endpts, params.era90_endpts, params.dtheta_fine)

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
    fig = plt.figure(figsize=(190 / 25.4, 230 / 25.4), dpi=params.dpi)
    axs = fig.subplots(4, 2)

    for row_idx, row in enumerate(axs):
        for col_idx, ax in enumerate(row):
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
            if col_idx == 0:
                ax.yaxis.set_major_locator(plticker.MultipleLocator(base=25.0))
            x_label = 'Percent Complete (%)' if row_idx == 3 else None
            y_label = 'Axial Rotation (Deg)' if col_idx == 0 else 'SPM{t}'
            style_axes(ax, x_label, y_label)

    mean_lns = []
    t_lns = []
    alpha_lns = []
    for idx_act, (activity, activity_df) in enumerate(db_er_endpts.groupby('Activity', observed=True)):
        for idx_traj, traj_name in enumerate(('ht', 'gh')):
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
            isb_ln = mean_sd_plot(axs[idx_act * 2 + idx_traj, 0], x, isb_mean, isb_sd,
                                  dict(color=color_map.colors[1], alpha=0.2),
                                  dict(color=color_map.colors[1], marker=markers[1], markevery=20))
            phadke_ln = mean_sd_plot(axs[idx_act * 2 + idx_traj, 0], x, phadke_mean, phadke_sd,
                                     dict(color=color_map.colors[7], alpha=0.2),
                                     dict(color=color_map.colors[7], marker=markers[3], markevery=20))
            isb_norm_ln = mean_sd_plot(axs[idx_act * 2 + idx_traj, 0], x, isb_norm_mean, isb_norm_sd,
                                       dict(color=color_map.colors[2], alpha=0.2),
                                       dict(color=color_map.colors[2], marker=markers[2], markevery=20))
            true_ln = mean_sd_plot(axs[idx_act * 2 + idx_traj, 0], x, true_mean, true_sd,
                                   dict(color=color_map.colors[0], alpha=0.2, hatch='...'),
                                   dict(color=color_map.colors[0], marker=markers[0], markevery=20))

            # spm
            isb_vs_true = spm_test(all_traj_isb[:, 1:],
                                   all_traj_true[:, 1:]).inference(alpha, two_tailed=True, **infer_params)
            isb_norm_vs_true = spm_test(all_traj_isb_norm[:, 1:],
                                        all_traj_true[:, 1:]).inference(alpha, two_tailed=True, **infer_params)
            phadke_vs_true = spm_test(all_traj_phadke[:, 1:],
                                      all_traj_true[:, 1:]).inference(alpha, two_tailed=True, **infer_params)

            # plot spm
            isb_true_t_ln, isb_true_alpha = spm_plot_alpha(axs[idx_act * 2 + idx_traj, 1], x[1:], isb_vs_true,
                                                           dict(color=color_map.colors[1], alpha=0.25),
                                                           dict(color=color_map.colors[1]))
            phadke_true_t_ln, phadke_true_alpha = spm_plot_alpha(axs[idx_act * 2 + idx_traj, 1], x[1:], phadke_vs_true,
                                                                 dict(color=color_map.colors[7], alpha=0.25),
                                                                 dict(color=color_map.colors[7]))
            isb_norm_true_t_ln, isb_norm_true_alpha = \
                spm_plot_alpha(axs[idx_act * 2 + idx_traj, 1], x[1:], isb_norm_vs_true,
                               dict(color=color_map.colors[2], alpha=0.25), dict(color=color_map.colors[2]))

            # print significance
            print('Activity: {} Joint: {}'.format(activity, traj_name.upper()))
            print('ISB vs True')
            print(extract_sig(isb_vs_true, x))
            print('ISB Rectified vs True')
            print(extract_sig(isb_norm_vs_true, x))
            print('Phadke vs True')
            print(extract_sig(phadke_vs_true, x))

            if idx_act == 0 and idx_traj == 0:
                # legend lines
                mean_lns.append(isb_ln[0])
                mean_lns.append(phadke_ln[0])
                mean_lns.append(isb_norm_ln[0])
                mean_lns.append(true_ln[0])

                t_lns.append(isb_true_t_ln[0])
                t_lns.append(phadke_true_t_ln[0])
                t_lns.append(isb_norm_true_t_ln[0])

                alpha_lns.append(isb_true_alpha)
                alpha_lns.append(phadke_true_alpha)
                alpha_lns.append(isb_norm_true_alpha)

    # figure title and legend
    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
    fig.suptitle('HT and GH Axial Rotation\nfor ERaR and ERa90', x=0.47, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.925)
    leg_left = fig.legend(mean_lns, ["yx'y''", "xz'y''", "yx'y'' Rectified", 'True'], loc='upper left',
                          bbox_to_anchor=(0, 1), ncol=2, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                          borderpad=0.2, labelspacing=0.4)
    leg_right = fig.legend(t_lns + [tuple(alpha_lns)],
                           ["yx'y''=True", "xz'y''=True", "yx'y'' Rectified=True", '$\\alpha=0.05$'],
                           loc='upper right', handler_map={tuple: HandlerTupleVertical(ndivide=None)},
                           bbox_to_anchor=(1, 1), ncol=2, handlelength=1.2, handletextpad=0.5, columnspacing=0.75,
                           labelspacing=0.3, borderpad=0.2)

    # add arrows indicating direction
    axs[0, 0].arrow(10, -76, 0, -40, length_includes_head=True, head_width=2, head_length=2)
    axs[0, 0].text(0, -70, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs[0, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'ERaR HT Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'ERaR GH Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[2, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'ERa90 HT Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[3, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'ERa90 GH Axial Rotation', ha='center', fontsize=11, fontweight='bold')
    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
