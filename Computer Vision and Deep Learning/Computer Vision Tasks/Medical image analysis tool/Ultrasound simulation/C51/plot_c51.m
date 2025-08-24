field_init(-1);
% Create the convex array (using your example)
%% System Parameters
f0 = 3e6;                    % Center frequency [Hz]
c = 1540;                    % Speed of sound [m/s]
lambda = c/f0;               % Wavelength [m], 1540/3e6 = 0.5133 mm
width = lambda/2;            % Element width, 0.5133/2 = 0.25665 mm

%% Transducer Geometry (C5-1 estimates)
element_height = 10/1000;    % Height of element [m], 10 mm
kerf = width/10;             % Kerf [m], 0.25665/10 = 0.025665 mm
focus = [0 0 48]/1000;       % Fixed focal point at 4.8 cm [m], 48 mm
N_elements = 160;            % Number of elements, 160
Rconvex = 55/1000;          % Convex Radius [m], 55 mm


Th = xdc_convex_array(N_elements, width, element_height, kerf, Rconvex, 1, 5, focus);

% Plot the transducer aperture
figure;
show_xdc(Th);
title('Convex Array Transducer');
xlabel('x [mm]');
ylabel('y [mm]');
zlabel('z [mm]');