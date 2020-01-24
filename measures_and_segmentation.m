%% Preprocessing
clc;

%% Coordinate Randomization
load("test_trajectory.mat")

%% Tests
[d, d_thresh, seg_thresh] = segmentation(x, y, z, 0.2, 3);
% disp(d);
fprintf("Distance Threshold: %d", d_thresh);
fprintf("\nSegmentation Threshold: %d ", seg_thresh);

%% Functions
function distance = path_length(x, y, z)
    x_sq_dist = diff(x) .^ 2;
    y_sq_dist = diff(y) .^ 2;
    z_sq_dist = diff(z) .^ 2;
    
    distance = 0;
    for i=1:length(x_sq_dist)
        distance = distance + ((x_sq_dist(i) + y_sq_dist(i) + z_sq_dist(i)) ^ 0.5);
    end
end

function vol = economy_of_volume(x, y, z, length_norm)
    vol = ((max(x) - min(x)) * (max(y) - min(y)) * (max(z) - min(z)))^(1/3);
    if length_norm
        vol = vol/path_length(x, y, z);
    end
end

function vol = geometric_mean_eigenvalues(x, y, z, length_norm)
     concat = vertcat(x, y, z);
     [~, ~, latent] = pca(concat);
     vol = prod(latent)^(1/3);
     if length_norm
         vol = vol/path_length(x, y, z);
     end    
end

function vol = arithmetic_mean_eigenvalues(x, y, z, length_norm)
    concat = vertcat(x, y, z);
    [~, ~, latent] = pca(concat);
    vol = mean(latent);
    if length_norm
        vol = num/path_length(x, y, z);
    end
end

function vol = max_of_eigenvalues(x, y, z, length_norm)
    concat = vertcat(x, y, z);
    [~, ~, latent] = pca(concat);
    vol = max(latent);
    if length_norm
        vol = num/path_length(x, y, z);
    end
end

function velocity = compute_velocity(x_coords, y_coords, z_coords, frame_rate)
    x_sq_dist = diff(x_coords) .^ 2;
    y_sq_dist = diff(y_coords) .^ 2;
    z_sq_dist = diff(z_coords) .^ 2;
    
    velocity = zeros(1, length(x_sq_dist));
    
    for i=1:length(velocity)
        velocity(i) = ((x_sq_dist(i) + y_sq_dist(i) + z_sq_dist(i))^0.5)/(1/frame_rate);
    end

end

function volume = space_occupancy(metric)
    if strcmp(metric, 'eov')
        volume = economy_of_volume(x, y, z, 1);
    elseif strcmp(metric, 'eov_nolen')
        volume = economy_of_volume(x, y, z, 0);
    elseif strcmp(metric, 'ge')
        volume = geometric_mean_eigenvalues(x, y, z, 1);
    elseif strcmp(metric, 'ge_nolen')
        volume = geometric_mean_eigenvalues(x, y, z, 0);
    elseif strcmp(metric, 'ae')
        volume = arithmetic_mean_eigenvalues(x, y, z, 1);
    elseif strcmp(metric, 'ae_nolen')
        volume = arithmetic_mean_eigenvalues(x, y, z, 0);
    elseif strcmp(metric, 'me')
        volume = max_of_eigenvalues(x, y, z, 1);
    elseif strcmp(metric, 'me_nolen')
        volume = max_of_eigenvalues(x, y, z, 0);
    else
        volume = -1;
    end
end

function [d_values, d_threshold, seg_index] = segmentation(x, y, z, alpha, med_filt_w_size)
    start = [x(1), y(1), z(1)];
    finish = [x(length(x)), y(length(y)), z(length(z))];
    denom = (finish(1) - start(1))^2 + (finish(2) - start(2))^2 + (finish(3) - start(3))^2;
    A = [(finish(1) - start(1))/denom, (finish(2) - start(2))/denom, (finish(3) - start(3))/denom];
    d = zeros(1, length(x));
    for i=1:length(x)
        d(i) = ((x(i) - start(1)) * A(1)) + ((y(i) - start(2)) * A(2)) + ((z(i) - start(3)) * A(3));
    end
    
    d_thresh = d(length(d))- alpha * d(length(d));
    d_thresh_indx = 0;
    for i=1:length(d)
       if d(i) > d_thresh
          d_thresh_indx = i;
          break;
       end
    end
    
    vel = compute_velocity(x, y, z, 30);
    vel = medfilt1(vel, med_filt_w_size);
    
    [maxVel, maxVelIndex] = max(vel);
    p1 = [maxVelIndex, maxVel, 0];
    p2 = [length(vel), vel(length(vel)), 0];
    orthog = zeros(1, length(vel));
    for i=1:length(vel)
       p3 = [i, vel(i), 0];
       val1 = cross(p2-p1, p3-p1);
       val1 = val1(3);
       val2 = norm(p2-p1);
       orthog(i) = abs(val1/val2);
    end
    
    [~, segmentation_index] = max(orthog(d_thresh_indx:end));
    segmentation_index = segmentation_index + d_thresh_indx;
    d_values = d;
    d_threshold = d_thresh_indx;
    seg_index = segmentation_index;
end