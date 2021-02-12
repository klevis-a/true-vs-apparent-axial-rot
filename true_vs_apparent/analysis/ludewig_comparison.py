"""Compare our dataset against Ludewig's.

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
ludewig_data: Path to files containing GH and ST trajectories for CA, SA, and FE from Ludewig et al.
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
    from true_vs_apparent.common.plot_utils import init_graphing, make_interactive, mean_sd_plot, style_axes
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db, extract_sub_rot, read_ludewig_data
    from true_vs_apparent.common.json_utils import get_params
    from true_vs_apparent.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare our dataset against Ludewig's", __package__, __file__))
    params = get_params(config_dir / 'parameters.json')

    # if the database variables have been retained (i.e. we are re-running the script) then skip retrieving data from
    # disk
    if not bool(distutils.util.strtobool(os.getenv('VARS_RETAINED', 'False'))):
        # ready db
        db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject, include_anthro=True)
        if params.excluded_trials:
            db = db[~db['Trial_Name'].str.contains('|'.join(params.excluded_trials))]
        db['Trial'].apply(pre_fetch)

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # read Ludewig's data
    ludewig_data = read_ludewig_data(params.ludewig_data)
    ludewig_ht = ludewig_data['gh']['ca']['HT_Elev'].to_numpy()

    # prepare database
    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, 'AC', params.dtheta_fine, params.dtheta_coarse,
               [ludewig_ht[0], ludewig_ht[-1]])

#%%
    # plot
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    ours_ht = db_elev.iloc[0]['traj_interp'].common_ht_range_coarse
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
    axs_std_hum[0, 0].arrow(20, -40, 0, -40, length_includes_head=True, head_width=2, head_length=2)
    axs_std_hum[0, 0].text(13, -40, 'Elevation', rotation=90, va='top', ha='left', fontsize=10)

    axs_std_hum[0, 0].arrow(35, -80, 0, 40, length_includes_head=True, head_width=2, head_length=2)
    axs_std_hum[0, 0].text(28, -45, 'Anterior', rotation=90, va='top', ha='left', fontsize=10)

    axs_std_hum[0, 1].arrow(20, -5, 0, -30, length_includes_head=True, head_width=2, head_length=2)
    axs_std_hum[0, 1].text(13, -8, 'External', rotation=90, va='top', ha='left', fontsize=10)

    # set axes limits
    for i in range(3):
        axs_std_hum[i, 0].set_ylim(-90, 40)
        axs_std_hum[i, 1].set_ylim(-90, 0)

    # plot our data
    leg_patch_ours_std_hum = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_elev = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 0]), axis=0)
        all_traj_poe = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 1]), axis=0)
        all_traj_axial = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 2]), axis=0)

        elev_mean = np.rad2deg(np.nanmean(all_traj_elev, axis=0))
        poe_mean = np.rad2deg(np.nanmean(all_traj_poe, axis=0))
        axial_mean = np.rad2deg(np.nanmean(all_traj_axial, axis=0))
        elev_sd = np.rad2deg(np.nanstd(all_traj_elev, ddof=1, axis=0))
        poe_sd = np.rad2deg(np.nanstd(all_traj_poe, ddof=1, axis=0))
        axial_sd = np.rad2deg(np.nanstd(all_traj_axial, ddof=1, axis=0))

        present_counts = np.count_nonzero(~np.isnan(all_traj_elev), axis=0)
        print(activity + ': ')
        print(present_counts)

        cur_row = act_row[activity.lower()]

        elev_ln = mean_sd_plot(axs_std_hum[cur_row, 0], ours_ht, elev_mean, elev_sd,
                               dict(color=color_map.colors[0], alpha=0.25),
                               dict(color=color_map.colors[0], marker=markers[0]))

        poe_ln = mean_sd_plot(axs_std_hum[cur_row, 0], ours_ht, poe_mean, poe_sd,
                              dict(color=color_map.colors[1], alpha=0.25),
                              dict(color=color_map.colors[1], marker=markers[1]))

        ax_ln = mean_sd_plot(axs_std_hum[cur_row, 1], ours_ht, axial_mean, axial_sd,
                             dict(color=color_map.colors[2], alpha=0.25),
                             dict(color=color_map.colors[2], marker=markers[2]))

        if idx == 0:
            leg_patch_ours_std_hum.append(elev_ln[0])
            leg_patch_ours_std_hum.append(poe_ln[0])
            leg_patch_ours_std_hum.append(ax_ln[0])

    # plot Ludewig's data
    leg_patch_ludewig_std_hum = []
    for idx, (motion_name, df) in enumerate(ludewig_data['gh'].items()):
        elev_ln = axs_std_hum[act_row[motion_name], 0].plot(ludewig_ht, df['Elevation'], color=color_map.colors[0],
                                                            marker=markers[0], ls='--', fillstyle='none')
        poe_ln = axs_std_hum[act_row[motion_name], 0].plot(ludewig_ht, df['PoE'], color=color_map.colors[1],
                                                           marker=markers[1], ls='--', fillstyle='none')
        ax_ln = axs_std_hum[act_row[motion_name], 1].plot(ludewig_ht, df['Axial'], color=color_map.colors[2],
                                                          marker=markers[2], ls='--', fillstyle='none')

        if idx == 0:
            leg_patch_ludewig_std_hum.append(elev_ln[0])
            leg_patch_ludewig_std_hum.append(poe_ln[0])
            leg_patch_ludewig_std_hum.append(ax_ln[0])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_std_hum.suptitle('Glenohumeral Motion Comparison', x=0.3, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    # fig_std.legend(list(itertools.chain(*zip(leg_patch_ours_std, leg_patch_ludewig_std))),
    #                ['Elevation', 'Elevation (Ludewig)', 'Plane of Elevation', 'Plane of Elevation (Ludewig)',
    #                 'Axial Rotation', 'Axial Rotation (Ludewig)'], loc='upper right', bbox_to_anchor=(1.01, 1.01))
    fig_std_hum.legend(leg_patch_ours_std_hum + leg_patch_ludewig_std_hum,
                       ['Elevation', 'Plane of Elevation', 'Axial Orientation', 'Ludewig', 'Ludewig', 'Ludewig'],
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

    # ############ AXIAL ROTATION COMPARISON FOR GH ##################################
    fig_axial = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_axial = fig_axial.subplots(2, 2)

    # style axes, add x and y labels
    style_axes(axs_axial[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_axial[1, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_axial[0, 1], None, None)
    style_axes(axs_axial[1, 1], 'Humerothoracic Elevation (Deg)', None)

    # add axes titles
    axs_axial[0, 0].set_title('Axial Rotation', y=0.9)
    axs_axial[0, 1].set_title('True Axial Rotation', y=0.9)
    axs_axial[1, 0].set_title('Axial Rotation', y=0.9)
    axs_axial[1, 1].set_title('True Axial Rotation', y=0.9)

    # set axes limits
    for i in range(2):
        axs_axial[i, 0].set_ylim(-50, 35)
        axs_axial[i, 1].set_ylim(-50, 35)

    leg_patch_ours_axial = []
    # plot our data
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_axial = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'euler.gh_phadke', 2]), axis=0)
        all_traj_true_axial = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_coarse_up', 'true_axial_rot', None]), axis=0)
        all_traj_axial = all_traj_axial - all_traj_axial[:, 0][..., np.newaxis]
        all_traj_true_axial = all_traj_true_axial - all_traj_true_axial[:, 0][..., np.newaxis]

        axial_mean = np.rad2deg(np.nanmean(all_traj_axial, axis=0))
        true_axial_mean = np.rad2deg(np.nanmean(all_traj_true_axial, axis=0))
        axial_sd = np.rad2deg(np.nanstd(all_traj_axial, ddof=1, axis=0))
        true_axial_sd = np.rad2deg(np.nanstd(all_traj_true_axial, ddof=1, axis=0))

        ax_ln = mean_sd_plot(axs_axial[0, 0], ours_ht, axial_mean, axial_sd,
                             dict(color=color_map.colors[idx], alpha=0.25),
                             dict(color=color_map.colors[idx], marker=markers[idx]))

        true_ax_ln = mean_sd_plot(axs_axial[0, 1], ours_ht, true_axial_mean, true_axial_sd,
                                  dict(color=color_map.colors[idx], alpha=0.25),
                                  dict(color=color_map.colors[idx], marker=markers[idx], ls='--', fillstyle='none'))

        leg_patch_ours_axial.append(ax_ln[0])
        leg_patch_ours_axial.append(true_ax_ln[0])

    # plot Ludewig's data
    leg_patch_ludewig_axial = []
    for idx, (motion_name, df) in enumerate(ludewig_data['gh'].items()):
        ax_ln = axs_axial[1, 0].plot(ludewig_ht, df['Axial'] - df['Axial'][0], color=color_map.colors[idx],
                                     marker=markers[idx])
        true_ax_ln = axs_axial[1, 1].plot(ludewig_ht, df['true_axial'], color=color_map.colors[idx],
                                          marker=markers[idx], ls='--', fillstyle='none')
        leg_patch_ludewig_axial.append(ax_ln[0])
        leg_patch_ludewig_axial.append(true_ax_ln[0])

    # figure title and axes legends
    plt.tight_layout(pad=0.25, h_pad=2.0, w_pad=0.5)
    fig_axial.suptitle('Glenohumeral Axial Rotation Comparison', y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    axs_axial[0, 0].legend(leg_patch_ours_axial[0::2], ['CA', 'SA', 'FE'], loc='lower left')
    axs_axial[1, 0].legend(leg_patch_ours_axial[0::2], ['CA', 'SA', 'FE'], loc='lower left')
    axs_axial[0, 1].legend(leg_patch_ours_axial[1::2], ['CA', 'SA', 'FE'], loc='lower left')
    axs_axial[1, 1].legend(leg_patch_ours_axial[1::2], ['CA', 'SA', 'FE'], loc='lower left')

    # add axes titles
    _, y0, _, h = axs_axial[0, 0].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.03, 'Current Investigation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[1, 0].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.02, 'Ludewig et al.', ha='center', fontsize=11, fontweight='bold')

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
    axs_std_scap[0, 0].arrow(115, -25, 0, 40, length_includes_head=True, head_width=2, head_length=2)
    axs_std_scap[0, 0].text(108, -30, 'Protraction', rotation=90, va='bottom', ha='left', fontsize=10)

    axs_std_scap[0, 0].arrow(132, 15, 0, -40, length_includes_head=True, head_width=2, head_length=2)
    axs_std_scap[0, 0].text(125, -25, 'Upward\nRotation', rotation=90, va='bottom', ha='center', fontsize=10)

    axs_std_scap[0, 1].arrow(20, 0, 0, 10, length_includes_head=True, head_width=2, head_length=2)
    axs_std_scap[0, 1].text(13, 1, 'Posterior', rotation=90, va='bottom', ha='left', fontsize=10)

    # set axes limits
    for i in range(3):
        axs_std_scap[i, 0].set_ylim(-70, 70)
        axs_std_scap[i, 1].set_ylim(-20, 10)

    # plot our data
    leg_patch_ours_std_scap = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_repro = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'euler.st_isb', 0]), axis=0)
        all_traj_latmed = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'euler.st_isb', 1]), axis=0)
        all_traj_tilt = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'euler.st_isb', 2]), axis=0)

        repro_mean = np.rad2deg(np.nanmean(all_traj_repro, axis=0))
        latmed_mean = np.rad2deg(np.nanmean(all_traj_latmed, axis=0))
        tilt_mean = np.rad2deg(np.nanmean(all_traj_tilt, axis=0))
        repro_sd = np.rad2deg(np.nanstd(all_traj_repro, ddof=1, axis=0))
        latmed_sd = np.rad2deg(np.nanstd(all_traj_latmed, ddof=1, axis=0))
        tilt_sd = np.rad2deg(np.nanstd(all_traj_tilt, ddof=1, axis=0))

        present_counts = np.count_nonzero(~np.isnan(all_traj_repro), axis=0)
        print(activity + ': ')
        print(present_counts)

        cur_row = act_row[activity.lower()]
        repro_ln = mean_sd_plot(axs_std_scap[cur_row, 0], ours_ht, repro_mean, repro_sd,
                                dict(color=color_map.colors[0], alpha=0.25),
                                dict(color=color_map.colors[0], marker=markers[0]))
        latmed_ln = mean_sd_plot(axs_std_scap[cur_row, 0], ours_ht, latmed_mean, latmed_sd,
                                 dict(color=color_map.colors[1], alpha=0.25),
                                 dict(color=color_map.colors[1], marker=markers[1]))
        tilt_ln = mean_sd_plot(axs_std_scap[cur_row, 1], ours_ht, tilt_mean, tilt_sd,
                               dict(color=color_map.colors[2], alpha=0.25),
                               dict(color=color_map.colors[2], marker=markers[2]))

        if idx == 0:
            leg_patch_ours_std_scap.append(repro_ln[0])
            leg_patch_ours_std_scap.append(latmed_ln[0])
            leg_patch_ours_std_scap.append(tilt_ln[0])

    # plot Ludewig's data
    leg_patch_ludewig_std_scap = []
    for idx, (motion_name, df) in enumerate(ludewig_data['st'].items()):
        repro_ln = axs_std_scap[act_row[motion_name], 0].plot(
            ludewig_ht, df['ReProtraction'], color=color_map.colors[0], marker=markers[0], ls='--', fillstyle='none')
        latmed_ln = axs_std_scap[act_row[motion_name], 0].plot(ludewig_ht, df['LatMedRot'], color=color_map.colors[1],
                                                               marker=markers[1], ls='--', fillstyle='none')
        tilt_ln = axs_std_scap[act_row[motion_name], 1].plot(ludewig_ht, df['Tilt'], color=color_map.colors[2],
                                                             marker=markers[2], ls='--', fillstyle='none')

        if idx == 0:
            leg_patch_ludewig_std_scap.append(repro_ln[0])
            leg_patch_ludewig_std_scap.append(latmed_ln[0])
            leg_patch_ludewig_std_scap.append(tilt_ln[0])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_std_scap.suptitle('Scapulothoracic Motion Comparison', x=0.3, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    fig_std_scap.legend(leg_patch_ours_std_scap + leg_patch_ludewig_std_scap,
                        ['Protraction', 'Upward Rotation', 'Tilt', 'Ludewig', 'Ludewig', 'Ludewig'],
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

    # ############ TILT COMPARISON FOR ST ##################################
    fig_tilt = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_tilt = fig_tilt.subplots(2, 2)

    # style axes, add x and y labels
    style_axes(axs_tilt[0, 0], None, 'Tilt (Deg)')
    style_axes(axs_tilt[1, 0], 'Humerothoracic Elevation (Deg)', 'Tilt (Deg)')
    style_axes(axs_tilt[0, 1], None, None)
    style_axes(axs_tilt[1, 1], 'Humerothoracic Elevation (Deg)', None)

    # add axes titles
    axs_tilt[0, 0].set_title('Tilt', y=0.9)
    axs_tilt[0, 1].set_title('True Tilt', y=0.9)
    axs_tilt[1, 0].set_title('Tilt', y=0.9)
    axs_tilt[1, 1].set_title('True Tilt', y=0.9)

    # set axes limits
    for i in range(2):
        axs_tilt[i, 0].set_ylim(-5, 20)
        axs_tilt[i, 1].set_ylim(-5, 20)

    leg_patch_ours_tilt = []
    # plot our data
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_tilt = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'euler.st_isb', 2]), axis=0)
        all_traj_true_tilt = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_coarse_up', 'true_axial_rot', None]), axis=0)
        all_traj_tilt = all_traj_tilt - all_traj_tilt[:, 0][..., np.newaxis]
        all_traj_true_tilt = all_traj_true_tilt - all_traj_true_tilt[:, 0][..., np.newaxis]

        tilt_mean = np.rad2deg(np.nanmean(all_traj_tilt, axis=0))
        true_tilt_mean = np.rad2deg(np.nanmean(all_traj_true_tilt, axis=0))
        tilt_sd = np.rad2deg(np.nanstd(all_traj_tilt, ddof=1, axis=0))
        true_tilt_sd = np.rad2deg(np.nanstd(all_traj_true_tilt, ddof=1, axis=0))

        tilt_ln = mean_sd_plot(axs_tilt[0, 0], ours_ht, tilt_mean, tilt_sd,
                               dict(color=color_map.colors[idx], alpha=0.25),
                               dict(color=color_map.colors[idx], marker=markers[idx]))
        true_tilt_ln = mean_sd_plot(axs_tilt[0, 1], ours_ht, true_tilt_mean, true_tilt_sd,
                                    dict(color=color_map.colors[idx], alpha=0.25),
                                    dict(color=color_map.colors[idx], marker=markers[idx]))

        leg_patch_ours_tilt.append(tilt_ln[0])
        leg_patch_ours_tilt.append(true_tilt_ln[0])

    # plot Ludewig's data
    leg_patch_ludewig_tilt = []
    for idx, (motion_name, df) in enumerate(ludewig_data['st'].items()):
        tilt_ln = axs_tilt[1, 0].plot(ludewig_ht, df['Tilt'] - df['Tilt'][0], color=color_map.colors[idx],
                                      marker=markers[idx])
        true_tilt_ln = axs_tilt[1, 1].plot(ludewig_ht, df['true_axial'], color=color_map.colors[idx],
                                           marker=markers[idx], ls='--', fillstyle='none')
        leg_patch_ludewig_tilt.append(tilt_ln[0])
        leg_patch_ludewig_tilt.append(true_tilt_ln[0])

    # figure title and axes legends
    plt.tight_layout(pad=0.25, h_pad=2.0, w_pad=0.5)
    fig_tilt.suptitle('Scapulothoracic Tilt Comparison', y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    axs_tilt[0, 0].legend(leg_patch_ours_axial[0::2], ['CA', 'SA', 'FE'], loc='upper left')
    axs_tilt[1, 0].legend(leg_patch_ours_axial[0::2], ['CA', 'SA', 'FE'], loc='upper left')
    axs_tilt[0, 1].legend(leg_patch_ours_axial[1::2], ['CA', 'SA', 'FE'], loc='upper left')
    axs_tilt[1, 1].legend(leg_patch_ours_axial[1::2], ['CA', 'SA', 'FE'], loc='upper left')

    # add axes titles
    _, y0, _, h = axs_axial[0, 0].get_position().bounds
    fig_tilt.text(0.5, y0 + h * 1.03, 'Current Investigation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[1, 0].get_position().bounds
    fig_tilt.text(0.5, y0 + h * 1.02, 'Ludewig et al.', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    plt.show()
