% setup the path for the toolbox, and check for updates
%
% ---


% add path using full path names
myname = mfilename('fullpath');

[pathstr,name,ext] = fileparts(myname);

gdir = [pathstr filesep 'src'];
ddir = [pathstr filesep 'demo'];

addpath(genpath(gdir))

% check for updates
%emhmm_check_updates()

% check for MEX files
% emhmm_mex_check()

% set global variables

% set default of color mode
global EMHMM_COLORPLOTS
EMHMM_COLORPLOTS = 1;