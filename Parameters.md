| Parameter Name       | Parameter Description                                        |
| -------------------- | ------------------------------------------------------------ |
| `logger_name`        | Name of the logger set up in logging.ini that will receive log messages. |
| `biplane_vicon_db_dir` | Path to the database folder of the associated data repository. |
| `excluded_trials`      | Trial names to exclude from analysis.                        |
| `scap_lateral`     | Landmark to utilize when defining the scapula's lateral (+Z) axis (AC, PLA, GC). |
| `torso_def`        | Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition. |
| `min_elev`         | Minimum HT elevation angle (in degrees) utilized for analysis that encompasses all trials. |
| `max_elev`         | Maximum HT elevation angle (in degrees) utilized for analysis that encompasses all trials. |
| `dtheta_coarse`    | Incremental angle (in degrees) to use for coarse interpolation between minimum and maximum HT elevation. For external rotation trials this parameter specifies the incremental percentage to utilize for interpolation. |
| `dtheta_fine`      | Incremental angle (in degrees) to use for fine interpolation between minimum and maximum HT elevation analyzed. For external rotation trials this parameter specifies the incremental percentage to utilize for interpolation. |
| `era90_endpts`     | Path to csv file containing start and stop frames (including both external and internal rotation) for external rotation in 90&deg; of abduction trials. A -1 for the start and stop frame of a trial indicates that the trial should be excluded. |
| `erar_endpts`      | Path to csv file containing start and stop frames (including both external and internal rotation) for external rotation in adduction trials. A -1 for the start and stop frame of a trial indicates that the trial should be excluded. |
| `decomp_method`    | The decomposition method for the sequence-specific humeral path (isb for y'x'y'' or phadke for xz'y''). |
| `trajectory`       | Whether to examine glenohumeral (gh) or humerothoracic (ht) angles. |
| `parametric`       | Whether to use a parametric (true) or non-parametric statistical test (false). |
| `ludewig_data`     | Path to files containing glenohumeral and scapulothoracic trajectories for CA, SA, and FE from Ludewig et al. See below for more details. |
| `trial_name`       | The trial to plot.                                           |
| `output_dir`       | Directory where PDF records should be output.                |
| `backend`          | Matplotlib backend to use for plotting (e.g. TkAgg, Qt5Agg, macosx, etc.). |
| `dpi`              | Dots (pixels) per inch for generated figure (e.g. 300).     |
| `fig_file`         | Path to file where to save figure. If empty string, then figure is shown on monitor. |

The `ludewig_data` parameter needs to specified in the following JSON format:

```
{
    "gh": {
      "ca": "data\\Ludewig_GH_CA_Raising.csv",
      "sa": "data\\Ludewig_GH_SA_Raising.csv",
      "fe": "data\\Ludewig_GH_FE_Raising.csv"
    },
    "st": {
      "ca": "data\\Ludewig_ST_CA_Raising.csv",
      "sa": "data\\Ludewig_ST_SA_Raising.csv",
      "fe": "data\\Ludewig_ST_FE_Raising.csv"
    }
 }
```

