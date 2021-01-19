"""Compare our dataset against Ludewig's.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
"""

import quaternion as q
from biokinepy.trajectory import PoseTrajectory


def create_gh_traj(df):
    q_elev = q.from_rotation_vector(np.deg2rad(df['Elevation'].to_numpy())[..., np.newaxis] * np.array([1, 0, 0]))
    q_poe = q.from_rotation_vector(np.deg2rad(df['PoE'].to_numpy())[..., np.newaxis] * np.array([0, 0, 1]))
    q_axial = q.from_rotation_vector(np.deg2rad(df['Axial'].to_numpy())[..., np.newaxis] * np.array([0, 1, 0]))

    quat_traj = q.as_float_array(q_elev * q_poe * q_axial)
    pos = np.zeros((q_elev.size, 3))
    traj = PoseTrajectory.from_quat(pos, quat_traj)
    traj.long_axis = np.array([0, 1, 0])
    return traj


def create_st_traj(df):
    q_repro = q.from_rotation_vector(np.deg2rad(df['ReProtraction'].to_numpy())[..., np.newaxis] * np.array([0, 1, 0]))
    q_latmed = q.from_rotation_vector(np.deg2rad(df['LatMedRot'].to_numpy())[..., np.newaxis] * np.array([1, 0, 0]))
    q_tilt = q.from_rotation_vector(np.deg2rad(df['Tilt'].to_numpy())[..., np.newaxis] * np.array([0, 0, 1]))

    quat_traj = q.as_float_array(q_repro * q_latmed * q_tilt)
    pos = np.zeros((q_repro.size, 3))
    traj = PoseTrajectory.from_quat(pos, quat_traj)
    traj.long_axis = np.array([0, 0, 1])
    return traj


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from true_vs_apparent.common.plot_utils import init_graphing, make_interactive, mean_sd_plot, style_axes
    from true_vs_apparent.common.database import create_db, BiplaneViconSubject, pre_fetch
    from true_vs_apparent.common.analysis_utils import prepare_db, extract_sub_rot
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
    convert_fnc = {'gh': create_gh_traj, 'st': create_st_traj}
    ludewig_data = {}
    for traj_name, motions in params.ludewig_data._asdict().items():
        ludewig_data[traj_name] = {}
        for motion_name, file_path in motions._asdict().items():
            df = pd.read_csv(file_path, dtype=np.float)
            ludewig_data[traj_name][motion_name] = df
            traj = convert_fnc[traj_name](df)
            df['true_axial'] = np.rad2deg(traj.true_axial_rot)

    # prepare database
    ludewig_ht = ludewig_data['gh']['ca']['HT_Elev'].to_numpy()
    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, True, params.dtheta_fine, params.dtheta_coarse,
               [ludewig_ht[0], ludewig_ht[-1]])

#%%
    # plot
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    ours_ht = db_elev.iloc[0]['traj_interp'].common_ht_range_coarse
    init_graphing(params.backend)
    plt.close('all')

    fig_axial = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs_axial = fig_axial.subplots(2, 2)

    # style axes, add x and y labels
    style_axes(axs_axial[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_axial[1, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_axial[0, 1], None, None)
    style_axes(axs_axial[1, 1], 'Humerothoracic Elevation (Deg)', None)

    # add axes titles
    axs_axial[0, 0].set_title("Apparent Axial Rotation (xz'y'')", y=0.9)
    axs_axial[0, 1].set_title('True Axial Rotation', y=0.9)
    axs_axial[1, 0].set_title("Apparent Axial Rotation (xz'y'')", y=0.9)
    axs_axial[1, 1].set_title('True Axial Rotation', y=0.9)

    # x ticks
    if int(ludewig_ht[0]) % 10 == 0:
        x_ticks_start = int(ludewig_ht[0])
    else:
        x_ticks_start = int(ludewig_ht[0]) + (10 - int(ludewig_ht[0]) % 10)

    if int(ludewig_ht[-1]) % 10 == 0:
        x_ticks_end = int(ludewig_ht[-1])
    else:
        x_ticks_end = int(ludewig_ht[-1]) + (10 - int(ludewig_ht[-1]) % 10)
    x_ticks = np.arange(x_ticks_start, x_ticks_end + 1, 20)
    x_ticks = np.unique(np.concatenate((x_ticks, np.array([ludewig_ht[0], ludewig_ht[-1]]))))

    # set axes limits and x-ticks
    for i in range(2):
        axs_axial[i, 0].set_ylim(-55, 35)
        axs_axial[i, 1].set_ylim(-55, 35)
        axs_axial[i, 0].set_xticks(x_ticks)
        axs_axial[i, 1].set_xticks(x_ticks)

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

        present_counts = np.count_nonzero(~np.isnan(all_traj_axial), axis=0)
        print(activity + ': ')
        print(present_counts)
        print('Apparent Axial Rotation Mean: {:.2f} True Axial Rotation Mean: {:.2f}'.
              format(axial_mean[-1], true_axial_mean[-1]))

        male_count = 0
        female_count = 0
        for tj_interp, gender in zip(activity_df['traj_interp'], activity_df['Gender']):
            if not np.isnan(tj_interp.ht.common_coarse_up.quat_float[0, 0]):
                if gender == 'F':
                    female_count += 1
                else:
                    male_count += 1

        print('Males: {}'.format(male_count))
        print('Females: {}'.format(female_count))

    # plot Ludewig's data
    leg_patch_ludewig_axial = []
    for idx, (motion_name, df) in enumerate(ludewig_data['gh'].items()):
        apparent_axial_norm = (df['Axial'] - df['Axial'][0]).to_numpy()
        ax_ln = axs_axial[1, 0].plot(ludewig_ht, apparent_axial_norm, color=color_map.colors[idx],
                                     marker=markers[idx])
        true_ax_ln = axs_axial[1, 1].plot(ludewig_ht, df['true_axial'], color=color_map.colors[idx],
                                          marker=markers[idx], ls='--', fillstyle='none')
        print(motion_name + ': ')
        print('Ludewig Apparent Axial Rotation Mean: {:.2f} True Axial Rotation Mean: {:.2f}'.
              format(apparent_axial_norm[-1], df['true_axial'].iloc[-1]))
        leg_patch_ludewig_axial.append(ax_ln[0])
        leg_patch_ludewig_axial.append(true_ax_ln[0])

    # figure title and axes legends
    plt.tight_layout(pad=0.25, h_pad=2.0, w_pad=0.5)
    fig_axial.suptitle('Apparent vs True GH Axial Rotation Across Studies', y=0.99, fontweight='bold')
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

    # add arrows indicating direction
    axs_axial[0, 0].arrow(25, 25, 0, -20, length_includes_head=True, head_width=2, head_length=2)
    axs_axial[0, 0].text(15, 25, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    make_interactive()

    if params.fig_file:
        fig_axial.savefig(params.fig_file)
    else:
        plt.show()
