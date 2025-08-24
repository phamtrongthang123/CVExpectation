field_init(-1);
% Basic parameters
f0 = 3.5e6;           % C5-1 center frequency [Hz]
fs = 100e6;           % Sampling frequency [Hz]
c = 1540;             % Speed of sound [m/s]

% Set Field II parameters
set_sampling(fs);
set_field('c', c);
set_field('att', 0.56);        % Attenuation [dB/(MHz*cm)]
set_field('att_f0', f0);       % Reference frequency

% Create Philips C5-1 curved transducer (approximate specs). https://chatgpt.com/share/68917895-3028-8012-a8f9-cde763663abf
N_elements = 160;              
pitch = 0.3469e-3;             
Rconvex = 60e-3;               
element_height = 12e-3;        
kerf = 0.05e-3;                

aperture = xdc_convex_array(N_elements, pitch, Rconvex, element_height, kerf, 1, 1, [0 0 0]);
xdc_focus(aperture, 0, [0 0 0.048]);  % Focus at 4.8 cm


%% Simulation 1: Phantom with 0.56 dB/cm/MHz attenuation
set_field('att', 0.56);
% Your simulation code for phantom 1
% Simulate ultrasound propagation with elevational blockage
[rf_data] = simulate_with_blockage(aperture, phantom, blockage_level);

% Step 2: Estimation (pretend we don't know the truth)
estimated_attenuation = estimate_attenuation_coefficient(rf_data);

% Estimation Bias = Estimated Value - True Value
% Step 3: Calculate bias
absolute_bias = estimated_attenuation - true_attenuation;
relative_bias = (estimated_attenuation - true_attenuation) / true_attenuation * 100;

% continue later:
% https://claude.ai/chat/98748067-7b8c-44f4-92f1-efea1b2cccd3 
% try to retry multiple times so that we can get an ensembled result