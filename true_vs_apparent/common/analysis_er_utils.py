from typing import NamedTuple, Callable, Union
import numpy as np
import pandas as pd
from true_vs_apparent.common.analysis_utils import get_trajs
from true_vs_apparent.common.interpolation import set_long_axis_hum


def interp_axial_rot_er(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int], extract_fnc: Callable,
                        delta: float = 0.1) -> np.ndarray:
    """Interpolate the axial rotation trajectory specified by traj_name (gh, ht, st), sub_rot (0, 1, 2, None),
    extract_fnc (extraction function), and delta percentage intervals between 0% (Start of Motion) and 100%
    (maximum axial rotation)."""
    y = extract_fnc(df_row, traj_name, sub_rot)
    start_idx = df_row['Start'] - 1
    stop_idx = df_row['Stop'] - 1
    y = y[start_idx:stop_idx+1]
    y = y - y[0]
    max_idx = np.argmax(np.absolute(y))
    frame_nums = np.arange(0, max_idx+1)
    num_frames = frame_nums[-1] - frame_nums[0] + 1
    # divide by (num_frames - 1) because I want the last frame to be 100
    frame_nums_norm = ((frame_nums - frame_nums[0]) / (num_frames - 1)) * 100
    desired_frames = np.arange(0, 100 + delta, delta)
    return np.interp(desired_frames, frame_nums_norm, y[0:max_idx+1], np.nan, np.nan)


def extract_true(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract HT or GH true axial rotation."""
    return df_row[traj_name].true_axial_rot


def extract_isb(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract HT or GH ISB axial rotation."""
    return df_row[traj_name].euler.yxy_intrinsic[:, 2]


def extract_phadke(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract HT or GH Phadke axial rotation."""
    return df_row[traj_name].euler.xzy_intrinsic[:, 2]


def extract_isb_norm(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract PoE Adjusted ISB axial rotation."""
    return df_row[traj_name].euler.yxy_intrinsic[:, 2] + df_row[traj_name].euler.yxy_intrinsic[:, 0]


def extract_st_induced(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract ST-induced true axial rotation."""
    return df_row[traj_name][:, sub_rot]


class PlotDirective(NamedTuple):
    """Class containing directives for what to plot.

    Attributes
    ----------
    traj: str
        Whether to plot HT, ST, or GH.
    extract_fnc: Callable
        Function that extracts the desired metric from traj.
    sub_rot: int, None
        Sub-rotation to extract from trajectory.
    y_label: str
        Y label for the plot.
    title: str
        Title string for the plot.
    """
    traj: str
    extract_fnc: Callable
    sub_rot: Union[int, None]
    y_label: str
    title: str


plot_directives = {
    'ht_true': PlotDirective('ht', extract_true, None, 'Axial Rotation (deg)', 'True HT Axial Rotation'),
    'ht_isb': PlotDirective('ht', extract_isb, None, 'Axial Rotation (deg)', "ISB HT (yx'y'') Axial Rotation"),
    'ht_isb_norm': PlotDirective('ht', extract_isb_norm, None, 'Axial Rotation (deg)',
                                 "Normalized ISB HT (yx'y'') Axial Rotation"),
    'ht_phadke': PlotDirective('ht', extract_phadke, None, 'Axial Rotation (deg)', "HT xz'y'' Axial Rotation"),
    'gh_true': PlotDirective('gh', extract_true, None, 'Axial Rotation (deg)', 'True GH Axial Rotation'),
    'gh_isb': PlotDirective('gh', extract_isb, None, 'Axial Rotation (deg)', "ISB GH (yx'y'') Axial Rotation"),
    'gh_isb_norm': PlotDirective('gh', extract_isb_norm, None, 'Axial Rotation (deg)',
                                 "Normalized ISB GH (yx'y'') Axial Rotation"),
    'gh_phadke': PlotDirective('gh', extract_phadke, None, 'Axial Rotation (deg)', "GH xz'y'' Axial Rotation")}


def ready_er_db(db: pd.DataFrame, torso_def: str, scap_lateral: str, erar_endpts_file: str, era90_endpts_file: str,
                dtheta_fine: float) -> pd.DataFrame:
    """Ready external rotation database for analysis."""
    db_er = db.loc[db['Trial_Name'].str.contains('_ERa90_|_ERaR_')].copy()
    db_er['ht'], db_er['gh'], db_er['st'] = \
        zip(*db_er['Trial'].apply(get_trajs, args=[db_er.attrs['dt'], torso_def, scap_lateral]))
    db_er['ht'].apply(set_long_axis_hum)
    db_er['gh'].apply(set_long_axis_hum)

    # add endpoints
    db_erar = db_er.loc[db_er['Trial_Name'].str.contains('_ERaR_')].copy()
    db_era90 = db_er.loc[db_er['Trial_Name'].str.contains('_ERa90_')].copy()
    erar_endpts = pd.read_csv(erar_endpts_file, index_col='Subject')
    era90_endpts = pd.read_csv(era90_endpts_file, index_col='Subject')
    db_erar_endpts = pd.merge(db_erar, erar_endpts, how='inner', left_on='Subject_Name', right_on='Subject',
                              left_index=False, right_index=True)
    db_era90_endpts = pd.merge(db_era90, era90_endpts, how='inner', left_on='Subject_Name', right_on='Subject',
                               left_index=False, right_index=True)
    db_er_endpts = pd.concat((db_erar_endpts, db_era90_endpts))
    db_er_endpts = db_er_endpts[db_er_endpts['Start'] != -1]

    for dir_name, plot_directive in plot_directives.items():
        db_er_endpts[dir_name] = db_er_endpts.apply(
            interp_axial_rot_er, args=[plot_directive.traj, plot_directive.sub_rot, plot_directive.extract_fnc,
                                       dtheta_fine], axis=1)

    return db_er_endpts
