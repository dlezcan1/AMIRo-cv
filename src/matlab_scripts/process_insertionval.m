%% process_insertionval.m
%
% this is a script to run through the data points and generate the FBG shape
% from measurements
% 
% - written by: Dimitri Lezcano

%% Set-up
% directories to iterate through
trial_dirs = dir("../../data/01-18-2021_Test-Insertion-Expmt/Insertion*/");
mask = strcmp({trial_dirs.name},".") | strcmp({trial_dirs.name}, "..");
trial_dirs = trial_dirs(~mask); % remove "." and ".." directories
trial_dirs = trial_dirs([trial_dirs.isdir]); % make sure all are directories

% files to find
fbgdata_file = "FBGdata_meanshift.xls";

% saving options
save_bool = false;

% directory separation
if ispc
    dir_sep = '\';
else
    dir_sep = '/';
end

% calibraiton matrices file
calib_dir = "../../data/01-18-2021_Test-Insertion-Expmt/";
calib_file = calib_dir + "needle_params-Jig_Calibration_11-15-20_weighted.json";


%% Load the calibration matrices and AA locations (from base)
fbgneedle = jsondecode(fileread(calib_file));

% AA parsing
num_aas = fbgneedle.x_ActiveAreas;
aa_base_locs = struct2array(fbgneedle.SensorLocations);
aa_tip_locs = fbgneedle.length - aa_base_locs;
cal_mats_cell = struct2cell(fbgneedle.CalibrationMatrices);
cal_mat_tensor = cat(3, cal_mats_cell{:});


%% Iterate through the files
for i = 1:length(trial_dirs)
    % trial operations
    L = str2double(trial_dirs(i).name);
    
    % trial directory
    d = strcat(trial_dirs(i).folder,dir_sep, trial_dirs(i).name, dir_sep);
    fbg_file = d + fbgdata_file;
    
    % load the fbg shift in
    wl_shift = readmatrix(fbg_file);
    wl_shift = reshape(wl_shift, [], 3)'; % reshape the array so AA's are across rows and Ch down columns
    
    % use calibration senssors
    curvatures = calibrate_fbgsensors(wl_shift, cal_mat_tensor);
    
end
