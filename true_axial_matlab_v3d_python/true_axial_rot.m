function axial_rot=true_axial_rot(time, frames)
% time holds the time (typically in seconds) for the supplied frames (must
% be length of n - see below).

% frames holds the rotation matrix trajectory for the GH or HT joint. 
% frames must be 3x3xn where n is the number of frames.

% This function assumes that the longitudinal axis is of the humerus  is 
% specified by the y-axis. If not, then frames(1:3,2,:) must be 
% appropriately adjusted. For example, if the z-axis specifies the
% longitudinal axis of the humerus then the appropriate computation is: 
% frames(1:3,3,:)
    ang_vel = computeAngVelocity(time, frames);
    axial_rot = cumtrapz(time,dot(squeeze(frames(1:3,2,:))',ang_vel,2));
end
