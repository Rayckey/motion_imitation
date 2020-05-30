close all;clear all;clc
syms x y z l1 l_upper l_lower
target_point = [x;y;z]; % (x,y,z) coordinates
l2 = norm(target_point(2:end));
phi1 = asin(l1/l2);
phi2 = atan(target_point(3)/target_point(2));
theta = phi1 + phi2;

T_ab = [ROTX(theta), zeros(3,1);zeros(1,3), 1]*[eye(3) [0;0;l1];zeros(1,3) 1];
new_target = inv(T_ab)*[target_point;1];

tip_2_hip_extent = -new_target(2);
tip_2_hip_swing = atan(new_target(1)/new_target(2));

upper_joint = asin(2*tip_2_hip_extent*cos(tip_2_hip_swing))-tip_2_hip_swing-pi/2;
lower_joint = pi/2+upper_joint-asin(4*tip_2_hip_extent-cos(upper_joint));