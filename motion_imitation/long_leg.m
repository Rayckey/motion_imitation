clear all;clc;close all;
dt = 0.01;
f_tg = 1;
phi_init = 0;
phi_prev = phi_init;
phi_t = phi_init;
phi_leg_diff = pi/2; %walking pattern; phase difference between legs
beta = 0.2; %proportion of air-time to ground-time
Cs = 0; %center for swing and extension
h_tg = 1; % walking height
alpha_tg = 45/180*pi; % swing amplitude
Ae = 0.5;
theta = 0;
kick_scale = 0.8; % smaller, the harder the legs kick


while phi_prev<6.28
    phi = mod(phi_prev + 2*pi*f_tg*dt,2*pi);
    phi_t = [phi_t phi];
    phi_prev = phi;
end
for i = 1:4
    phi_leg(i,:) = mod(phi_t + (phi_leg_diff*i-pi/2),2*pi);
end

for i = 1:4
    for j = 1:length(phi_leg)
        if phi_leg(i,j)<2*pi*beta && phi_leg(i,j) >= 0
            t_prime(i,j) = phi_leg(i,j)/2/beta;
            h_tg(i,j) = 1;
        else
            t_prime(i,j) = 2*pi-(2*pi-phi_leg(i,j))/(2*(1-beta));
            h_tg(i,j) = -kick_scale*Ae*sin(t_prime(i,j))+1;
        end
    end
end

u_tg_1 = [Cs + alpha_tg*cos(t_prime(1,:));h_tg(1,:) + Ae*sin(t_prime(1,:)) + theta*cos(t_prime(1,:))];
u_tg_2 = [Cs + alpha_tg*cos(t_prime(2,:));h_tg(2,:) + Ae*sin(t_prime(2,:)) + theta*cos(t_prime(2,:))];
u_tg_3 = [Cs + alpha_tg*cos(t_prime(3,:));h_tg(3,:) + Ae*sin(t_prime(3,:)) + theta*cos(t_prime(3,:))];
u_tg_4 = [Cs + alpha_tg*cos(t_prime(4,:));h_tg(4,:) + Ae*sin(t_prime(4,:)) + theta*cos(t_prime(4,:))];
figure(1)
hold on
axis([-0.5 0.5 0.5 1.5])

for i = 1:length(t_prime)
    subplot(2,2,1)
    hold on
    axis([-0.5 0.5 0.5 1.5])
    plot(tan(u_tg_1(1,i))*Ae,u_tg_1(2,i),'b*')
    subplot(2,2,2)
    axis([-0.5 0.5 0.5 1.5])
    plot(tan(u_tg_2(1,i))*Ae,u_tg_2(2,i),'b*')
    hold on
    subplot(2,2,3)
    axis([-0.5 0.5 0.5 1.5])
    plot(tan(u_tg_3(1,i))*Ae,u_tg_3(2,i),'b*')
    hold on

    subplot(2,2,4)
    axis([-0.5 0.5 0.5 1.5])
    plot(tan(u_tg_4(1,i))*Ae,u_tg_4(2,i),'b*')
    hold on
    pause(0.01)
end
