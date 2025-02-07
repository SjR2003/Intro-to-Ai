%% Generater data
num = 1;               
den = [1 2 1];         
G = tf(num, den);

delay = 0.5;           
[num_delay, den_delay] = pade(delay, 1); 
G_delay = tf(num_delay, den_delay) * G;

variance = 0.7;
noise = sqrt(variance) * randn(1000, 1); 

t = 0:0.01:10;  
u = sin(t);  
y = lsim(G_delay, u, t) + noise';  

figure;
margin(G_delay); 
%% PID
Ku = 10;              
Tu = 1.2;             

Kp = 0.21 * Ku;
Ti = 1.4 * Tu;
Td = 0.4 * Tu;
Ki = Kp / Ti;
Kd = Kp * Td;

PID = pid(Kp, Ki, Kd);

T_pid = feedback(PID * G_delay, 1);

figure;
step(T_pid);
title('Step response');

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



