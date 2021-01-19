"""Compare using GC vs AC when analyzing trials. All components of GH and ST trajectories are compared, but I am mainly
interested in GH axial rotation.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
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

    config_dir = Path(mod_arg_parser('Compare using GC vs AC when analyzing trials', __package__, __file__))
    params = get_params(config_dir / 'parameters.json')

    # if the database variables have been retained (i.e. we are re-running the script) then skip retrieving data from
    # disk
    if not bool(distutils.util.strtobool(os.getenv('VARS_RETAINED', 'False'))):
        # ready db
        db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject, include_anthro=True)
        if params.excluded_trials:
            db = db[~db['Trial_Name'].str.contains('|'.join(params.excluded_trials))]
        db['Trial'].apply(pre_fetch)

    # relevant parameters
    output_path = Path(params.output_dir)

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # prepare database
    db_elev_ac = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    db_elev_gc = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev_ac, params.torso_def, True, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    prepare_db(db_elev_gc, params.torso_def, False, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])

#%%
    # plot
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}
    ours_ht = db_elev_ac.iloc[0]['traj_interp'].common_ht_range_coarse

    init_graphing(params.backend)
    plt.close('all')

    # ############ EULER ANGLE COMPARISONS FOR GH ##################################
    fig_std_hum = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_std_hum = fig_std_hum.subplots(3, 2)

    # style axes, add x and y labels
    style_axes(axs_std_hum[0, 0], None, 'Elevation/PoE (Deg)')
    style_axes(axs_std_hum[1, 0], None, 'Elevation/PoE (Deg)')
    style_axes(axs_std_hum[2, 0], 'Humerothoracic Elevation (Deg)', 'Elevation/PoE (Deg)')
    style_axes(axs_std_hum[0, 1], None, 'Axial Orientation (Deg)')
    style_axes(axs_std_hum[1, 1], None, 'Axial Orientation (Deg)')
    style_axes(axs_std_hum[2, 1], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')

    # add arrows indicating direction
    axs_std_hum[0, 0].arrow(30, -60, 0, -40, length_includes_head=True, head_width=2, head_length=2)
    axs_std_hum[0, 0].text(23, -55, 'Elevation', rotation=90, va='top', ha='left', fontsize=10)

    axs_std_hum[0, 0].arrow(45, -100, 0, 40, length_includes_head=True, head_width=2, head_length=2)
    axs_std_hum[0, 0].text(38, -60, 'Anterior', rotation=90, va='top', ha='left', fontsize=10)

    axs_std_hum[0, 1].arrow(30, -20, 0, -30, length_includes_head=True, head_width=2, head_length=2)
    axs_std_hum[0, 1].text(23, -22, 'External', rotation=90, va='top', ha='left', fontsize=10)

    # set axes limits
    for i in range(3):
        axs_std_hum[i, 0].set_ylim(-120, 40)
        axs_std_hum[i, 1].set_ylim(-85, -5)

    # plot our data
    leg_patch_ours_std_hum = []
    for idx, ((activity, activity_df_ac), (_, activity_df_gc)) in \
            enumerate(zip(db_elev_ac.groupby('Activity', observed=True), db_elev_gc.groupby('Activity', observed=True))):
        all_traj_elev_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 0]), axis=0)
        all_traj_poe_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 1]), axis=0)
        all_traj_axial_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 2]), axis=0)

        all_traj_elev_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 0]), axis=0)
        all_traj_poe_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 1]), axis=0)
        all_traj_axial_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 2]), axis=0)

        elev_mean_ac = np.rad2deg(np.mean(all_traj_elev_ac, axis=0))
        poe_mean_ac = np.rad2deg(np.mean(all_traj_poe_ac, axis=0))
        axial_mean_ac = np.rad2deg(np.mean(all_traj_axial_ac, axis=0))
        elev_sd_ac = np.rad2deg(np.std(all_traj_elev_ac, ddof=1, axis=0))
        poe_sd_ac = np.rad2deg(np.std(all_traj_poe_ac, ddof=1, axis=0))
        axial_sd_ac = np.rad2deg(np.std(all_traj_axial_ac, ddof=1, axis=0))

        elev_mean_gc = np.rad2deg(np.mean(all_traj_elev_gc, axis=0))
        poe_mean_gc = np.rad2deg(np.mean(all_traj_poe_gc, axis=0))
        axial_mean_gc = np.rad2deg(np.mean(all_traj_axial_gc, axis=0))
        elev_sd_gc = np.rad2deg(np.std(all_traj_elev_gc, ddof=1, axis=0))
        poe_sd_gc = np.rad2deg(np.std(all_traj_poe_gc, ddof=1, axis=0))
        axial_sd_gc = np.rad2deg(np.std(all_traj_axial_gc, ddof=1, axis=0))

        present_counts = np.count_nonzero(~np.isnan(all_traj_elev_ac), axis=0)
        print(activity + ': ')
        print(present_counts)

        cur_row = act_row[activity.lower()]
        elev_ln_ac = mean_sd_plot(axs_std_hum[cur_row, 0], ours_ht, elev_mean_ac, elev_sd_ac,
                                  dict(color=color_map.colors[0], alpha=0.25),
                                  dict(color=color_map.colors[0], marker=markers[0], ls='--', fillstyle='none'))
        poe_ln_ac = mean_sd_plot(axs_std_hum[cur_row, 0], ours_ht, poe_mean_ac, poe_sd_ac,
                                 dict(color=color_map.colors[1], alpha=0.25),
                                 dict(color=color_map.colors[1], marker=markers[1], ls='--', fillstyle='none'))
        ax_ln_ac = mean_sd_plot(axs_std_hum[cur_row, 1], ours_ht, axial_mean_ac, axial_sd_ac,
                                dict(color=color_map.colors[2], alpha=0.25),
                                dict(color=color_map.colors[2], marker=markers[2], ls='--', fillstyle='none'))

        elev_ln_gc = mean_sd_plot(axs_std_hum[cur_row, 0], ours_ht, elev_mean_gc, elev_sd_gc,
                                  dict(color=color_map.colors[3], alpha=0.25),
                                  dict(color=color_map.colors[3], marker=markers[0]))
        poe_ln_gc = mean_sd_plot(axs_std_hum[cur_row, 0], ours_ht, poe_mean_gc, poe_sd_gc,
                                 dict(color=color_map.colors[4], alpha=0.25),
                                 dict(color=color_map.colors[4], marker=markers[1]))
        ax_ln_gc = mean_sd_plot(axs_std_hum[cur_row, 1], ours_ht, axial_mean_gc, axial_sd_gc,
                                dict(color=color_map.colors[5], alpha=0.25),
                                dict(color=color_map.colors[5], marker=markers[2]))

        if idx == 0:
            leg_patch_ours_std_hum.append(elev_ln_gc[0])
            leg_patch_ours_std_hum.append(poe_ln_gc[0])
            leg_patch_ours_std_hum.append(ax_ln_gc[0])
            leg_patch_ours_std_hum.append(elev_ln_ac[0])
            leg_patch_ours_std_hum.append(poe_ln_ac[0])
            leg_patch_ours_std_hum.append(ax_ln_ac[0])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_std_hum.suptitle('Glenohumeral Motion Comparison', x=0.3, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_std_hum.legend(leg_patch_ours_std_hum,
                       ['Elevation (GC)', 'Plane of Elevation (GC)', 'Axial Orientation (GC)', 'AC', 'AC', 'AC'],
                       loc='upper right', bbox_to_anchor=(1.01, 1.01), ncol=2, handlelength=2.0, handletextpad=0.5,
                       columnspacing=0.75)

    # add axes titles
    _, y0, _, h = axs_std_hum[0, 0].get_position().bounds
    fig_std_hum.text(0.5, y0 + h * 1.05, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_std_hum[1, 0].get_position().bounds
    fig_std_hum.text(0.5, y0 + h * 1.05, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_std_hum[2, 0].get_position().bounds
    fig_std_hum.text(0.5, y0 + h * 1.05, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    # ############ EULER ANGLE COMPARISONS FOR ST ##################################
    fig_std_scap = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_std_scap = fig_std_scap.subplots(3, 2)

    # style axes, add x and y labels
    style_axes(axs_std_scap[0, 0], None, 'Protraction/Upward Rotation (Deg)')
    style_axes(axs_std_scap[1, 0], None, 'Protraction/Upward Rotation (Deg)')
    style_axes(axs_std_scap[2, 0], 'Humerothoracic Elevation (Deg)', 'Protraction/Upward Rotation (Deg)')
    style_axes(axs_std_scap[0, 1], None, 'Tilt (Deg)')
    style_axes(axs_std_scap[1, 1], None, 'Tilt (Deg)')
    style_axes(axs_std_scap[2, 1], 'Humerothoracic Elevation (Deg)', 'Tilt (Deg)')

    # add arrows indicating direction
    axs_std_scap[0, 0].arrow(132, -15, 0, 40, length_includes_head=True, head_width=2, head_length=2)
    axs_std_scap[0, 0].text(125, -20, 'Protraction', rotation=90, va='bottom', ha='left', fontsize=10)

    axs_std_scap[0, 0].arrow(120, 25, 0, -40, length_includes_head=True, head_width=2, head_length=2)
    axs_std_scap[0, 0].text(113, 32, 'Upward\nRotation', rotation=90, va='top', ha='center', fontsize=10)

    axs_std_scap[0, 1].arrow(30, -2, 0, 7, length_includes_head=True, head_width=2, head_length=2)
    axs_std_scap[0, 1].text(23, -2, 'Posterior', rotation=90, va='bottom', ha='left', fontsize=10)

    # set axes limits
    for i in range(3):
        axs_std_scap[i, 0].set_ylim(-70, 70)
        axs_std_scap[i, 1].set_ylim(-20, 5)

    # plot our data
    leg_patch_ours_std_scap = []
    for idx, ((activity, activity_df_ac), (_, activity_df_gc)) in \
            enumerate(zip(db_elev_ac.groupby('Activity', observed=True), db_elev_gc.groupby('Activity', observed=True))):
        all_traj_repro_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'euler.st_isb', 0]), axis=0)
        all_traj_latmed_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'euler.st_isb', 1]), axis=0)
        all_traj_tilt_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'euler.st_isb', 2]), axis=0)

        all_traj_repro_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'euler.st_isb', 0]), axis=0)
        all_traj_latmed_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'euler.st_isb', 1]), axis=0)
        all_traj_tilt_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'euler.st_isb', 2]), axis=0)

        repro_mean_ac = np.rad2deg(np.mean(all_traj_repro_ac, axis=0))
        latmed_mean_ac = np.rad2deg(np.mean(all_traj_latmed_ac, axis=0))
        tilt_mean_ac = np.rad2deg(np.mean(all_traj_tilt_ac, axis=0))
        repro_sd_ac = np.rad2deg(np.std(all_traj_repro_ac, ddof=1, axis=0))
        latmed_sd_ac = np.rad2deg(np.std(all_traj_latmed_ac, ddof=1, axis=0))
        tilt_sd_ac = np.rad2deg(np.std(all_traj_tilt_ac, ddof=1, axis=0))

        repro_mean_gc = np.rad2deg(np.mean(all_traj_repro_gc, axis=0))
        latmed_mean_gc = np.rad2deg(np.mean(all_traj_latmed_gc, axis=0))
        tilt_mean_gc = np.rad2deg(np.mean(all_traj_tilt_gc, axis=0))
        repro_sd_gc = np.rad2deg(np.std(all_traj_repro_gc, ddof=1, axis=0))
        latmed_sd_gc = np.rad2deg(np.std(all_traj_latmed_gc, ddof=1, axis=0))
        tilt_sd_gc = np.rad2deg(np.std(all_traj_tilt_gc, ddof=1, axis=0))

        present_counts = np.count_nonzero(~np.isnan(all_traj_repro_ac), axis=0)
        print(activity + ': ')
        print(present_counts)

        cur_row = act_row[activity.lower()]
        repro_ln_ac = mean_sd_plot(axs_std_scap[cur_row, 0], ours_ht, repro_mean_ac, repro_sd_ac,
                                   dict(color=color_map.colors[0], alpha=0.25),
                                   dict(color=color_map.colors[0], marker=markers[0]))
        latmed_ln_ac = mean_sd_plot(axs_std_scap[cur_row, 0], ours_ht, latmed_mean_ac, latmed_sd_ac,
                                    dict(color=color_map.colors[1], alpha=0.25),
                                    dict(color=color_map.colors[1], marker=markers[1]))
        tilt_ln_ac = mean_sd_plot(axs_std_scap[cur_row, 1], ours_ht, tilt_mean_ac, tilt_sd_ac,
                                  dict(color=color_map.colors[2], alpha=0.25),
                                  dict(color=color_map.colors[2], marker=markers[2]))

        repro_ln_gc = mean_sd_plot(axs_std_scap[cur_row, 0], ours_ht, repro_mean_gc, repro_sd_gc,
                                   dict(color=color_map.colors[3], alpha=0.25),
                                   dict(color=color_map.colors[3], marker=markers[0]))
        latmed_ln_gc = mean_sd_plot(axs_std_scap[cur_row, 0], ours_ht, latmed_mean_gc, latmed_sd_gc,
                                    dict(color=color_map.colors[4], alpha=0.25),
                                    dict(color=color_map.colors[4], marker=markers[1]))
        tilt_ln_gc = mean_sd_plot(axs_std_scap[cur_row, 1], ours_ht, tilt_mean_gc, tilt_sd_gc,
                                  dict(color=color_map.colors[5], alpha=0.25),
                                  dict(color=color_map.colors[5], marker=markers[2]))

        if idx == 0:
            leg_patch_ours_std_scap.append(repro_ln_ac[0])
            leg_patch_ours_std_scap.append(latmed_ln_ac[0])
            leg_patch_ours_std_scap.append(tilt_ln_ac[0])
            leg_patch_ours_std_scap.append(repro_ln_gc[0])
            leg_patch_ours_std_scap.append(latmed_ln_gc[0])
            leg_patch_ours_std_scap.append(tilt_ln_gc[0])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_std_scap.suptitle('Scapulothoracic Motion Comparison', x=0.3, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_std_scap.legend(leg_patch_ours_std_scap, ['Protraction', 'Upward Rotation', 'Tilt', 'AC', 'AC', 'AC'],
                        loc='upper right', bbox_to_anchor=(1.01, 1.01), ncol=2, handlelength=2.0, handletextpad=0.5,
                        columnspacing=0.75)

    # add axes titles
    _, y0, _, h = axs_std_scap[0, 0].get_position().bounds
    fig_std_scap.text(0.5, y0 + h * 1.05, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_std_scap[1, 0].get_position().bounds
    fig_std_scap.text(0.5, y0 + h * 1.05, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_std_scap[2, 0].get_position().bounds
    fig_std_scap.text(0.5, y0 + h * 1.05, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    # ############ AXIAL ROTATION COMPARISON FOR GH ##################################
    fig_axial = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_axial = fig_axial.subplots(3, 2)

    # style axes, add x and y labels
    style_axes(axs_axial[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_axial[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_axial[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_axial[0, 1], None, 'True Axial Rotation (Deg)')
    style_axes(axs_axial[1, 1], None, 'True Axial Rotation (Deg)')
    style_axes(axs_axial[2, 1], 'Humerothoracic Elevation (Deg)', 'True Axial Rotation (Deg)')

    # set axes limits
    for i in range(2):
        axs_axial[i, 0].set_ylim(-50, 35)
        axs_axial[i, 1].set_ylim(-50, 35)

    leg_patch_ours_axial = []
    # plot our data
    for idx, ((activity, activity_df_ac), (_, activity_df_gc)) in \
            enumerate(zip(db_elev_ac.groupby('Activity', observed=True), db_elev_gc.groupby('Activity', observed=True))):
        all_traj_axial_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 2]), axis=0)
        all_traj_true_axial_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'true_axial_rot', None]), axis=0)
        all_traj_axial_ac = all_traj_axial_ac - all_traj_axial_ac[:, 0][..., np.newaxis]
        all_traj_true_axial_ac = all_traj_true_axial_ac - all_traj_true_axial_ac[:, 0][..., np.newaxis]

        all_traj_axial_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 2]), axis=0)
        all_traj_true_axial_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'true_axial_rot', None]), axis=0)
        all_traj_axial_gc = all_traj_axial_gc - all_traj_axial_gc[:, 0][..., np.newaxis]
        all_traj_true_axial_gc = all_traj_true_axial_gc - all_traj_true_axial_gc[:, 0][..., np.newaxis]

        axial_mean_ac = np.rad2deg(np.mean(all_traj_axial_ac, axis=0))
        true_axial_mean_ac = np.rad2deg(np.mean(all_traj_true_axial_ac, axis=0))
        axial_sd_ac = np.rad2deg(np.std(all_traj_axial_ac, ddof=1, axis=0))
        true_axial_sd_ac = np.rad2deg(np.std(all_traj_true_axial_ac, ddof=1, axis=0))

        axial_mean_gc = np.rad2deg(np.mean(all_traj_axial_gc, axis=0))
        true_axial_mean_gc = np.rad2deg(np.mean(all_traj_true_axial_gc, axis=0))
        axial_sd_gc = np.rad2deg(np.std(all_traj_axial_gc, ddof=1, axis=0))
        true_axial_sd_gc = np.rad2deg(np.std(all_traj_true_axial_gc, ddof=1, axis=0))

        cur_row = act_row[activity.lower()]
        ax_ln_ac = mean_sd_plot(axs_axial[cur_row, 0], ours_ht, axial_mean_ac, axial_sd_ac,
                                dict(color=color_map.colors[0], alpha=0.25),
                                dict(color=color_map.colors[0], marker=markers[1]))
        true_ax_ln_ac = mean_sd_plot(axs_axial[cur_row, 1], ours_ht, true_axial_mean_ac, true_axial_sd_ac,
                                     dict(color=color_map.colors[4], alpha=0.25),
                                     dict(color=color_map.colors[4], marker=markers[2]))

        ax_ln_gc = mean_sd_plot(axs_axial[cur_row, 0], ours_ht, axial_mean_gc, axial_sd_gc,
                                dict(color=color_map.colors[1], alpha=0.25),
                                dict(color=color_map.colors[1], marker=markers[1], ls='--', fillstyle='none'))
        true_ax_ln_gc = mean_sd_plot(axs_axial[cur_row, 1], ours_ht, true_axial_mean_gc, true_axial_sd_gc,
                                     dict(color=color_map.colors[5], alpha=0.25),
                                     dict(color=color_map.colors[5], marker=markers[2], ls='--', fillstyle='none'))

        leg_patch_ours_axial.append(ax_ln_ac[0])
        leg_patch_ours_axial.append(true_ax_ln_ac[0])
        leg_patch_ours_axial.append(ax_ln_gc[0])
        leg_patch_ours_axial.append(true_ax_ln_gc[0])

    # figure title and axes legends
    plt.tight_layout(pad=0.25, h_pad=2.0, w_pad=0.5)
    fig_axial.suptitle('Glenohumeral Axial Rotation Comparison', x=0.3, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_axial.legend(leg_patch_ours_axial, ['Axial Rotation', 'True Axial Rotation', 'AC', 'AC'],
                     loc='upper right', bbox_to_anchor=(1.01, 1.01), ncol=2, handlelength=2.0, handletextpad=0.5,
                     columnspacing=0.75)

    # add axes titles
    _, y0, _, h = axs_axial[0, 0].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.03, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[1, 0].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[2, 0].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.02, 'Forward Elevation Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    # ############ AXIAL ROTATION COMPARISON FOR GH NORMALIZED BY START ##################################
    fig_axial_start = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_axial_start = fig_axial_start.subplots(3, 2)

    # style axes, add x and y labels
    style_axes(axs_axial_start[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_axial_start[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_axial_start[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_axial_start[0, 1], None, 'True Axial Rotation (Deg)')
    style_axes(axs_axial_start[1, 1], None, 'True Axial Rotation (Deg)')
    style_axes(axs_axial_start[2, 1], 'Humerothoracic Elevation (Deg)', 'True Axial Rotation (Deg)')

    # set axes limits
    # for i in range(2):
    #     axs_axial[i, 0].set_ylim(-50, 35)
    #     axs_axial[i, 1].set_ylim(-50, 35)

    leg_patch_ours_axial_start = []
    # plot our data
    for idx, ((activity, activity_df_ac), (_, activity_df_gc)) in \
            enumerate(zip(db_elev_ac.groupby('Activity', observed=True), db_elev_gc.groupby('Activity', observed=True))):
        all_traj_axial_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true_axial_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_coarse_up', 'true_axial_rot', None, 'up']), axis=0)

        all_traj_axial_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_true_axial_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_coarse_up', 'true_axial_rot', None, 'up']), axis=0)

        axial_mean_ac = np.rad2deg(np.mean(all_traj_axial_ac, axis=0))
        true_axial_mean_ac = np.rad2deg(np.mean(all_traj_true_axial_ac, axis=0))
        axial_sd_ac = np.rad2deg(np.std(all_traj_axial_ac, ddof=1, axis=0))
        true_axial_sd_ac = np.rad2deg(np.std(all_traj_true_axial_ac, ddof=1, axis=0))

        axial_mean_gc = np.rad2deg(np.mean(all_traj_axial_gc, axis=0))
        true_axial_mean_gc = np.rad2deg(np.mean(all_traj_true_axial_gc, axis=0))
        axial_sd_gc = np.rad2deg(np.std(all_traj_axial_gc, ddof=1, axis=0))
        true_axial_sd_gc = np.rad2deg(np.std(all_traj_true_axial_gc, ddof=1, axis=0))

        cur_row = act_row[activity.lower()]
        ax_ln_ac = mean_sd_plot(axs_axial_start[cur_row, 0], ours_ht, axial_mean_ac, axial_sd_ac,
                                dict(color=color_map.colors[0], alpha=0.25),
                                dict(color=color_map.colors[0], marker=markers[1]))
        true_ax_ln_ac = mean_sd_plot(axs_axial_start[cur_row, 1], ours_ht, true_axial_mean_ac, true_axial_sd_ac,
                                     dict(color=color_map.colors[4], alpha=0.25),
                                     dict(color=color_map.colors[4], marker=markers[2]))

        ax_ln_gc = mean_sd_plot(axs_axial_start[cur_row, 0], ours_ht, axial_mean_gc, axial_sd_gc,
                                dict(color=color_map.colors[1], alpha=0.25),
                                dict(color=color_map.colors[1], marker=markers[1], ls='--', fillstyle='none'))
        true_ax_ln_gc = mean_sd_plot(axs_axial_start[cur_row, 1], ours_ht, true_axial_mean_gc, true_axial_sd_gc,
                                     dict(color=color_map.colors[5], alpha=0.25),
                                     dict(color=color_map.colors[5], marker=markers[2], ls='--', fillstyle='none'))

        leg_patch_ours_axial_start.append(ax_ln_ac[0])
        leg_patch_ours_axial_start.append(true_ax_ln_ac[0])
        leg_patch_ours_axial_start.append(ax_ln_gc[0])
        leg_patch_ours_axial_start.append(true_ax_ln_gc[0])

    # figure title and axes legends
    plt.tight_layout(pad=0.25, h_pad=2.0, w_pad=0.5)
    fig_axial_start.suptitle('GH Axial Rotation Comparison Norm Start', x=0.3, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_axial_start.legend(leg_patch_ours_axial, ['Axial Rotation', 'True Axial Rotation', 'AC', 'AC'],
                           loc='upper right', bbox_to_anchor=(1.01, 1.01), ncol=2, handlelength=2.0, handletextpad=0.5,
                           columnspacing=0.75)

    # add axes titles
    _, y0, _, h = axs_axial_start[0, 0].get_position().bounds
    fig_axial_start.text(0.5, y0 + h * 1.03, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial_start[1, 0].get_position().bounds
    fig_axial_start.text(0.5, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial_start[2, 0].get_position().bounds
    fig_axial_start.text(0.5, y0 + h * 1.02, 'Forward Elevation Plane Abduction', ha='center', fontsize=11,
                         fontweight='bold')

    make_interactive()

    # ############ AXIAL ROTATION SPM COMPARISON FOR GH ##################################
    x = db_elev_ac.iloc[0]['traj_interp'].common_ht_range_fine
    alpha = 0.05
    fig_axial_spm = plt.figure(figsize=(270 / 25.4, 190 / 25.4))
    axs_axial_spm = fig_axial_spm.subplots(3, 3)

    # style axes, add x and y labels
    style_axes(axs_axial_spm[0, 0], None, 'SPM{t}')
    style_axes(axs_axial_spm[1, 0], None, 'SPM{t}')
    style_axes(axs_axial_spm[2, 0], 'Humerothoracic Elevation (Deg)', 'SPM{t}')
    style_axes(axs_axial_spm[0, 1], None, None)
    style_axes(axs_axial_spm[1, 1], None, None)
    style_axes(axs_axial_spm[2, 1], 'Humerothoracic Elevation (Deg)', None)
    style_axes(axs_axial_spm[0, 2], None, None)
    style_axes(axs_axial_spm[1, 2], None, None)
    style_axes(axs_axial_spm[2, 2], 'Humerothoracic Elevation (Deg)', None)

    leg_patch_axial_spm = []
    # plot our data
    for idx, ((activity, activity_df_ac), (_, activity_df_gc)) in \
            enumerate(zip(db_elev_ac.groupby('Activity', observed=True), db_elev_gc.groupby('Activity', observed=True))):
        if activity_df_ac.empty:
            continue
        all_traj_axial_ac = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2]), axis=0)
        all_traj_axial_ac_norm_start = np.stack(activity_df_ac['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_axial_ac_norm = all_traj_axial_ac - all_traj_axial_ac[:, 0][..., np.newaxis]

        all_traj_axial_gc = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2]), axis=0)
        all_traj_axial_gc_norm_start = np.stack(activity_df_gc['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_phadke', 2, 'up']), axis=0)
        all_traj_axial_gc_norm = all_traj_axial_gc - all_traj_axial_gc[:, 0][..., np.newaxis]

        gc_vs_ac_orient = spm1d.stats.ttest_paired(all_traj_axial_ac, all_traj_axial_gc).inference(alpha,
                                                                                                   two_tailed=True)
        gc_vs_ac_rot = spm1d.stats.ttest_paired(all_traj_axial_ac_norm[:, 1:],
                                                all_traj_axial_gc_norm[:, 1:]).inference(alpha, two_tailed=True)
        gc_vs_ac_rot_norm_start = spm1d.stats.ttest_paired(
            all_traj_axial_ac_norm_start, all_traj_axial_gc_norm_start).inference(alpha, two_tailed=True)

        # plot spm
        cur_row = act_row[activity.lower()]
        orient_ln = spm_plot(axs_axial_spm[cur_row, 0], x, gc_vs_ac_orient, dict(color=color_map.colors[0], alpha=0.25),
                             dict(color=color_map.colors[0]))
        rotation_ln = spm_plot(axs_axial_spm[cur_row, 1], x[1:], gc_vs_ac_rot,
                               dict(color=color_map.colors[1], alpha=0.25),
                               dict(color=color_map.colors[1]))
        rotation_start_ln = spm_plot(axs_axial_spm[cur_row, 2], x, gc_vs_ac_rot_norm_start,
                                     dict(color=color_map.colors[2], alpha=0.25),
                                     dict(color=color_map.colors[2]))

        leg_patch_axial_spm.append(orient_ln[0])
        leg_patch_axial_spm.append(rotation_start_ln[0])
        leg_patch_axial_spm.append(rotation_ln[0])

    # figure title and axes legends
    plt.tight_layout(pad=0.25, h_pad=2.0, w_pad=0.5)
    fig_axial_spm.suptitle('Glenohumeral Axial Rotation SPM Comparison', x=0.3, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_axial_spm.legend(leg_patch_axial_spm, ['Axial Orientation', 'Rotation Norm Start SPM{t}', 'Rotation SPM{t}'],
                         loc='upper right', bbox_to_anchor=(1.01, 1.01), ncol=2, handlelength=2.0, handletextpad=0.5,
                         columnspacing=0.75)

    # add axes titles
    _, y0, _, h = axs_axial_spm[0, 0].get_position().bounds
    fig_axial_spm.text(0.5, y0 + h * 1.03, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial_spm[1, 0].get_position().bounds
    fig_axial_spm.text(0.5, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial_spm[2, 0].get_position().bounds
    fig_axial_spm.text(0.5, y0 + h * 1.02, 'Forward Elevation Plane Abduction', ha='center', fontsize=11,
                       fontweight='bold')

    make_interactive()

    plt.show()
