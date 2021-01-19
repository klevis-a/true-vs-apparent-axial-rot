from typing import Union, Tuple, Sequence
import pandas as pd
import numpy as np
import quaternion as q
from biokinepy.cs import ht_inv
from biokinepy.segment_cs import scapula_cs_isb
from biokinepy.trajectory import PoseTrajectory
from true_vs_apparent.common.database import trajectories_from_trial, BiplaneViconTrial
from true_vs_apparent.common.interpolation import ShoulderTrajInterp, interp_vec_traj
from true_vs_apparent.common.up_down import analyze_up_down
from true_vs_apparent.common.python_utils import rgetattr
import logging

log = logging.getLogger(__name__)


def get_trajs(trial: BiplaneViconTrial, dt: float, torso_def: str, use_ac: bool = False) \
        -> Tuple[PoseTrajectory, PoseTrajectory, PoseTrajectory]:
    """Return HT, GH, and ST trajectory from a given trial."""
    torso, scap, hum = trajectories_from_trial(trial, dt, torso_def=torso_def)
    # this is not efficient because I recompute the scapula CS for each trial, but it can be done at the subject level
    # however, in the grand scheme of things, this computation is trivial
    if use_ac:
        scap_gc = scapula_cs_isb(trial.subject.scapula_landmarks['GC'], trial.subject.scapula_landmarks['IA'],
                                 trial.subject.scapula_landmarks['TS'])
        scap_ac = scapula_cs_isb(trial.subject.scapula_landmarks['AC'], trial.subject.scapula_landmarks['IA'],
                                 trial.subject.scapula_landmarks['TS'])
        scap_gc_ac = ht_inv(scap_gc) @ scap_ac
        scap = PoseTrajectory.from_ht(scap.ht @ scap_gc_ac[np.newaxis, ...], dt=scap.dt, frame_nums=scap.frame_nums)

    ht = hum.in_trajectory(torso)
    gh = hum.in_trajectory(scap)
    st = scap.in_trajectory(torso)
    return ht, gh, st


def extract_sub_rot(shoulder_traj: ShoulderTrajInterp, traj_def: str, y_def: str, decomp_method: str,
                    sub_rot: Union[int, None]) -> np.ndarray:
    """Extract the specified subrotation given an interpolated shoulder trajectory (shoulder_traj),
    joint (traj_def, e.g. ht, gh, st), interpolation (y_def, e.g. common_fine_up),
    decomposition method (decomp_method, e.g. euler.ht_isb), and subrotation (sub_rot, e.g. 0, 1, 2, None)."""
    # first extract ht, gh, or st
    joint_traj = getattr(shoulder_traj, traj_def)
    # Then extract the decomp_method. Note that each JointTrajectory actually computes separate scalar interpolation for
    # true_axial_rot (that's why I don't access the PoseTrajectory below) because true_axial_rot is path dependent so
    # it doesn't make sense to compute it on a trajectory that starts at 25 degrees (for example)
    if decomp_method == 'true_axial_rot':
        y = getattr(joint_traj, 'axial_rot_' + y_def)
    elif decomp_method == 'induced_axial_rot':
        y = getattr(joint_traj, 'induced_axial_rot_' + y_def)[:, sub_rot]
    else:
        y = rgetattr(getattr(joint_traj, y_def), decomp_method)[:, sub_rot]
    return y


def extract_sub_rot_norm(shoulder_traj: ShoulderTrajInterp, traj_def: str, y_def: str, decomp_method: str,
                         sub_rot: Union[int, None], norm_by: str) -> np.ndarray:
    """Extract and normalize the specified subrotation given an interpolated shoulder trajectory (shoulder_traj),
    joint (traj_def, e.g. ht, gh, st), interpolation (y_def, e.g. common_fine_up),
    decomposition method (decomp_method, e.g. euler.ht_isb), subrotation (sub_rot, e.g. 0, 1, 2, None), and
    normalization section (norm_by, e.g. traj, up, down, ...)."""
    y = extract_sub_rot(shoulder_traj, traj_def, y_def, decomp_method, sub_rot)
    # first extract ht, gh, or st
    joint_traj = getattr(shoulder_traj, traj_def)
    if decomp_method == 'true_axial_rot':
        if norm_by == 'traj':
            y0 = getattr(joint_traj, 'axial_rot')[0]
        else:
            y0 = getattr(joint_traj, 'axial_rot_' + norm_by)[0]
    elif decomp_method == 'induced_axial_rot':
        if norm_by == 'traj':
            y0 = getattr(joint_traj, 'induced_axial_rot')[0, sub_rot]
        else:
            y0 = getattr(joint_traj, 'induced_axial_rot_' + norm_by)[0, sub_rot]
    else:
        y0 = rgetattr(getattr(joint_traj, norm_by), decomp_method)[0, sub_rot]

    return y-y0


def ht_min_max(df_row: pd.Series) -> Tuple[float, float, float, int]:
    """Compute minimum HT elevation during elevation and depression, as well as maximum HT elevation and the frame index
    where it occurs."""
    max_elev = df_row['ht'].euler.ht_isb[:, 1].min()
    max_elev_idx = np.argmin(df_row['ht'].euler.ht_isb[:, 1])
    min_elev_elev = df_row['ht'].euler.ht_isb[:max_elev_idx+1, 1].max()
    min_elev_depress = df_row['ht'].euler.ht_isb[max_elev_idx:, 1].max()
    return min_elev_elev, min_elev_depress, max_elev, max_elev_idx


def prepare_db(db: pd.DataFrame, torso_def: str, use_ac: bool, dtheta_fine: float, dtheta_coarse: float,
               ht_endpts_common: Union[Sequence, np.ndarray], should_fill: bool = True,
               should_clean: bool = True) -> None:
    """Prepared database for analysis by computing HT, ST, and GH trajectories; excluding and filling frames;
    determining trajectory endpoints and interpolating."""
    def create_interp_traj(df_row, fine_dt, coarse_dt, common_ht_endpts):
        return ShoulderTrajInterp(df_row['Trial_Name'], df_row['ht'], df_row['gh'], df_row['st'],
                                  df_row['up_down_analysis'], fine_dt, coarse_dt, common_ht_endpts)

    db['ht'], db['gh'], db['st'] = zip(*db['Trial'].apply(get_trajs, args=[db.attrs['dt'], torso_def, use_ac]))
    if should_clean:
        clean_up_trials(db)
    if should_fill:
        fill_trials(db)
    db['up_down_analysis'] = db['ht'].apply(analyze_up_down)
    db['traj_interp'] = db.apply(create_interp_traj, axis=1, args=[dtheta_fine, dtheta_coarse, ht_endpts_common])


def clean_up_trials(db: pd.DataFrame) -> None:
    """Exclude frames from trials where subject has not gotten ready to perform thumb-up elevation yet."""
    # the subject is axially rotating their arm to prepare for a thumb up arm elevation in the first 47 frames for SA,
    # and first 69 frames for CA
    clean_up_tasks = {'N022_SA_t01': 47, 'N022_CA_t01': 69}
    for trial_name, start_frame in clean_up_tasks.items():
        ht = db.loc[trial_name, 'ht']
        gh = db.loc[trial_name, 'gh']
        st = db.loc[trial_name, 'st']
        db.loc[trial_name, 'ht'] = PoseTrajectory.from_ht(ht.ht[start_frame:, :, :], ht.dt, ht.frame_nums[start_frame:])
        db.loc[trial_name, 'gh'] = PoseTrajectory.from_ht(gh.ht[start_frame:, :, :], gh.dt, gh.frame_nums[start_frame:])
        db.loc[trial_name, 'st'] = PoseTrajectory.from_ht(st.ht[start_frame:, :, :], st.dt, st.frame_nums[start_frame:])


def fill_trials(db: pd.DataFrame) -> None:
    """Fill trials."""
    def create_quat(angle, axis):
        return q.from_float_array(np.concatenate((np.array([np.cos(angle/2)]), np.sin(angle/2) * axis)))

    def fill_traj(traj, frames_to_avg, frames_to_fill):
        dt = traj.dt

        # compute averages
        ang_vel_avg_up = np.mean(traj.ang_vel[0:frames_to_avg, :], axis=0)
        ang_vel_avg_up_angle = np.linalg.norm(ang_vel_avg_up)
        ang_vel_avg_up_axis = ang_vel_avg_up / ang_vel_avg_up_angle
        ang_vel_avg_down = np.mean(traj.ang_vel[-frames_to_avg:, :], axis=0)
        ang_vel_avg_down_angle = np.linalg.norm(ang_vel_avg_down)
        ang_vel_avg_down_axis = ang_vel_avg_down / ang_vel_avg_down_angle
        vel_avg_up = np.mean(traj.vel[0:frames_to_avg, :], axis=0)
        vel_avg_down = np.mean(traj.vel[-frames_to_avg:, :], axis=0)

        # add additional frames
        pos_up_filled = np.stack([vel_avg_up * dt * i + traj.pos[0] for i in range(-frames_to_fill, 0)], 0)
        pos_down_filled = np.stack([vel_avg_down * dt * i + traj.pos[-1] for i in range(1, frames_to_fill + 1)], 0)
        quat_up_filled = np.stack([create_quat(ang_vel_avg_up_angle * dt * i, ang_vel_avg_up_axis) * traj.quat[0]
                                  for i in range(-frames_to_fill, 0)], 0)
        quat_down_filled = np.stack([create_quat(ang_vel_avg_down_angle * dt * i, ang_vel_avg_down_axis) * traj.quat[-1]
                                    for i in range(1, frames_to_fill + 1)], 0)

        # create new trajectory
        new_frame_nums = np.concatenate((np.arange(traj.frame_nums[0] - frames_to_fill, traj.frame_nums[0]),
                                         traj.frame_nums,
                                         np.arange(traj.frame_nums[-1] + 1, traj.frame_nums[-1] + frames_to_fill + 1)))
        if new_frame_nums[0] < 0:
            new_frame_nums = new_frame_nums + (-new_frame_nums[0])

        pos = np.concatenate((pos_up_filled, traj.pos, pos_down_filled), axis=0)
        quat = q.as_float_array(np.concatenate((quat_up_filled, traj.quat, quat_down_filled), axis=0))

        return PoseTrajectory.from_quat(pos, quat, dt, new_frame_nums)

    # this trial is extremely close to reaching the 25 deg ht elevation mark both up (25.28) and down (26.17), so I have
    # elected to fill it because it will give us this datapoint for the rest of the trials
    db.loc['N003A_SA_t01', 'ht'] = fill_traj(db.loc['N003A_SA_t01', 'ht'], 5, 5)
    db.loc['N003A_SA_t01', 'gh'] = fill_traj(db.loc['N003A_SA_t01', 'gh'], 5, 5)
    db.loc['N003A_SA_t01', 'st'] = fill_traj(db.loc['N003A_SA_t01', 'st'], 5, 5)


def create_gh_traj_ludewig(df):
    q_elev = q.from_rotation_vector(np.deg2rad(df['Elevation'].to_numpy())[..., np.newaxis] * np.array([1, 0, 0]))
    q_poe = q.from_rotation_vector(np.deg2rad(df['PoE'].to_numpy())[..., np.newaxis] * np.array([0, 0, 1]))
    q_axial = q.from_rotation_vector(np.deg2rad(df['Axial'].to_numpy())[..., np.newaxis] * np.array([0, 1, 0]))

    quat_traj = q.as_float_array(q_elev * q_poe * q_axial)
    pos = np.zeros((q_elev.size, 3))
    traj = PoseTrajectory.from_quat(pos, quat_traj)
    traj.long_axis = np.array([0, 1, 0])
    return traj


def create_st_traj_ludewig(df):
    q_repro = q.from_rotation_vector(np.deg2rad(df['ReProtraction'].to_numpy())[..., np.newaxis] * np.array([0, 1, 0]))
    q_latmed = q.from_rotation_vector(np.deg2rad(df['LatMedRot'].to_numpy())[..., np.newaxis] * np.array([1, 0, 0]))
    q_tilt = q.from_rotation_vector(np.deg2rad(df['Tilt'].to_numpy())[..., np.newaxis] * np.array([0, 0, 1]))

    quat_traj = q.as_float_array(q_repro * q_latmed * q_tilt)
    pos = np.zeros((q_repro.size, 3))
    traj = PoseTrajectory.from_quat(pos, quat_traj)
    traj.long_axis = np.array([0, 0, 1])
    return traj


def read_ludewig_data(ludewig_files):
    convert_fnc = {'gh': create_gh_traj_ludewig, 'st': create_st_traj_ludewig}
    ludewig_data = {}
    for traj_name, motions in ludewig_files._asdict().items():
        ludewig_data[traj_name] = {}
        for motion_name, file_path in motions._asdict().items():
            df = pd.read_csv(file_path, dtype=np.float)
            ludewig_data[traj_name][motion_name] = df
            traj = convert_fnc[traj_name](df)
            df['true_axial'] = np.rad2deg(traj.true_axial_rot)

    return ludewig_data
