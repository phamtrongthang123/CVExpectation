% Simplified Field II simulation for Philips C5-1 curved array
% Based on the original linear array example, modified for curved geometry
%
% This script assumes that the field_init procedure has been called

%% System Parameters (based on C5-1 specifications)
f0 = 3e6;                    % Center frequency [Hz] 
fs = 100e6;                  % Sampling frequency [Hz]
c = 1540;                    % Speed of sound [m/s]
lambda = c/f0;               % Wavelength [m]
width = lambda/2;            % Element width (typical for curved arrays)

%% Transducer Geometry (C5-1 estimates)
element_height = 10/1000;    % Height of element [m] (typical for abdominal)
kerf = width/10;             % Kerf [m] (smaller for curved arrays)
focus = [0 0 48]/1000;       % Fixed focal point at 4.8 cm [m] (from your paper)
N_elements = 160;            % Number of elements (C5-1 specification)
N_active = 64;               % Active elements 
Rconvex = 55/1000;          % Radius of curvature [m] (estimated)

%% Set Field II sampling frequency
set_sampling(fs);

%% Generate CURVED apertures (key change from original)
emit_aperture = xdc_convex_array(N_elements, width, element_height, kerf, Rconvex, 1, 5, focus);
receive_aperture = xdc_convex_array(N_elements, width, element_height, kerf, Rconvex, 1, 5, focus);

%% Simple impulse response (keep basic)
impulse_response=sin(2*pi*f0*(0:1/fs:2/f0));
impulse_response=impulse_response.*hanning(max(size(impulse_response)))';
xdc_impulse(emit_aperture, impulse_response);
xdc_impulse(receive_aperture, impulse_response);

%% Load phantom (reduce complexity for speed)
[phantom_positions, phantom_amplitudes] = cyst_phantom(10000); % Reduced from 10000

%% Imaging parameters
no_lines = N_elements - N_active + 1;  % Number of scan lines
dx = width;                            % Lateral spacing
z_focus = 48/1000;                    % Focus depth (from your paper)

%% Pre-allocate storage
image_data = zeros(1, no_lines);    % Estimate max size

%% Main Imaging Loop
for i=1:no_lines
 i
 % Find position for imaging
 x=(i-1-no_lines/2)*dx;
 % Dynamic Focusing: Set the focus for this direction
 xdc_center_focus (emit_aperture, [x 0 0]);
 xdc_focus (emit_aperture, 0, [x 0 z_focus]);
 xdc_center_focus (receive_aperture, [x 0 0]);
 xdc_focus (receive_aperture, 0, [x 0 z_focus]);
 % Dynamic Apodization: Set the active elements using the apodization
 apo=[zeros(1, i-1) hamming(N_active)' zeros(1, N_elements-N_active-i+1)];
 xdc_apodization (emit_aperture, 0, apo);
 xdc_apodization (receive_aperture, 0, apo);
 % Field Calculation: Calculate the received response
 [v, t1]=calc_scat(emit_aperture, receive_aperture, phantom_positions, phantom_amplitudes);
 % Store the result
 image_data(1:max(size(v)),i)=v;
 times(i) = t1;
end

%% Clean up apertures
xdc_free(emit_aperture);
xdc_free(receive_aperture);

% Adjust the data in time and display it as
% a gray scale image
min_sample=min(times)*fs;
for i=1:no_lines
 rf_env=abs(hilbert([zeros(round(times(i)*fs-min_sample),1); image_data(:,i)]));
 env(1:size(rf_env,1),i)=rf_env;
end

% make logarithmic compression to a 60 dB dynamic range
% with proper units on the axis
env_dB=20*log10(env);
env_dB=env_dB-max(max(env_dB));
env_gray=127*(env_dB+60)/60;
depth=((0:size(env,1)-1)+min_sample)/fs*c/2;
x=((1:no_lines)-no_lines/2)*dx;
image(x*1000, depth*1000, env_gray)
xlabel('Lateral distance [mm]')
ylabel('Depth [mm]')
axis('image')
colormap(gray(128))
title('Image of cyst phantom (60 dB dynamic range)')
