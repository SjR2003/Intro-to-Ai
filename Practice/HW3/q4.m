%% Generate data
numerator = [1 0];  % s = [1 0]
denominator = [1 1]; % s + 1 = [1 1]

% Create the transfer function
G = tf(numerator, denominator);    

% Add delay using Pade approximation
delay = 0.5;           
[num_delay, den_delay] = pade(delay, 1); 
G_delay = tf(num_delay, den_delay) * G;

% Add noise to the output
variance = 0.7;
t = 0:0.01:10;  
noise = sqrt(variance) * randn(length(t), 1); 

% Define input signal
u = sin(t);  
y = lsim(G_delay, u, t) + noise';  

% Plot the Bode plot with margin
figure;
margin(G_delay); 
title('Bode Plot with Delay');

%% PID Tuning using Ziegler-Nichols Method

% Define a range of Kp values to test
Kp_values = 0.1:0.1:10; 

% Initialize variables
Ku = 0;
Tu = 0;

% Find Ku and Tu
for Kp = Kp_values
    C = pid(Kp); 
    closed_loop = feedback(C * G_delay, 1);
    
    [y_step, t_step] = step(closed_loop);
    
    % Find oscillations using peaks
    [pks, locs] = findpeaks(y_step, t_step);
    
    if length(pks) > 2  % Ensure multiple peaks for periodicity
        Ku = Kp;
        Tu = mean(diff(locs)); % Average time between peaks
        break;
    end
end

fprintf('Critical Gain (Ku): %.4f\n', Ku);
fprintf('Critical Period (Tu): %.4f\n', Tu);

% Calculate PID gains using Ziegler-Nichols rules
Kp = 0.6 * Ku; 
Ti = 0.5 * Tu; 
Td = 0.125 * Tu; 

Ki = Kp / Ti; 
Kd = Kp * Td; 

% Open PID tuner
% pidTuner(G_delay, 'pid');

% Create the PID controller
PID = pid(Kp, Ki, Kd);

% Closed-loop system with PID controller
T_pid = feedback(PID * G_delay, 1);

% Plot the step response
figure;
step(T_pid);
title('Step Response with Ziegler-Nichols PID Controller');
xlabel('Time');
ylabel('Amplitude');
grid on;


%% make fuzzy fis
fis = mamfis('Name', 'FuzzyPID');

fis = addInput(fis, [-1 1], 'Name', 'Error');
fis = addMF(fis, 'Error', 'gaussmf', [0.1 0], 'Name', 'Zero');
fis = addMF(fis, 'Error', 'gaussmf', [0.3 -0.5], 'Name', 'Negative');
fis = addMF(fis, 'Error', 'gaussmf', [0.3 0.5], 'Name', 'Positive');

fis = addInput(fis, [-1 1], 'Name', 'DeltaError');
fis = addMF(fis, 'DeltaError', 'gaussmf', [0.1 0], 'Name', 'Zero');
fis = addMF(fis, 'DeltaError', 'gaussmf', [0.3 -0.5], 'Name', 'Negative');
fis = addMF(fis, 'DeltaError', 'gaussmf', [0.3 0.5], 'Name', 'Positive');

fis = addOutput(fis, [0 4], 'Name', 'Kp');
fis = addMF(fis, 'Kp', 'trimf', [0 1 2], 'Name', 'Low');
fis = addMF(fis, 'Kp', 'trimf', [1 2 3], 'Name', 'Medium');
fis = addMF(fis, 'Kp', 'trimf', [2 3 4], 'Name', 'High');

fis = addOutput(fis, [0 4], 'Name', 'Ki');
fis = addMF(fis, 'Ki', 'trimf', [0 1 2], 'Name', 'Low');
fis = addMF(fis, 'Ki', 'trimf', [1 2 3], 'Name', 'Medium');
fis = addMF(fis, 'Ki', 'trimf', [2 3 4], 'Name', 'High');

fis = addOutput(fis, [0 4], 'Name', 'Kd');
fis = addMF(fis, 'Kd', 'trimf', [0 1 2], 'Name', 'Low');
fis = addMF(fis, 'Kd', 'trimf', [1 2 3], 'Name', 'Medium');
fis = addMF(fis, 'Kd', 'trimf', [2 3 4], 'Name', 'High');

ruleList = [
    1 1 3 2 1 1 1;
    2 2 2 2 2 1 1;
    3 3 1 1 3 1 1
];
fis = addRule(fis, ruleList);
writeFIS(fis, 'FuzzyPID.fis');

%% Fuzzy-PID
fis = readfis('FuzzyPID.fis');

error = 0.1;          
delta_error = 0.01;   

params = evalfis(fis, [error, delta_error]);
Kp_fuzzy = params(1);
Ki_fuzzy = params(2);
Kd_fuzzy = params(3);

PID_fuzzy = pid(Kp_fuzzy, Ki_fuzzy, Kd_fuzzy);

T_fuzzy = feedback(PID_fuzzy * G_delay, 1);

figure;
step(T_fuzzy);
title('Step Response - Fuzzy PID');

%% Compare 
figure;
step(T_pid, 'b', T_fuzzy, 'r');
legend('Classical PID', 'Fuzzy PID');
title('Comparison of Step Response: Classical vs. Fuzzy PID');



