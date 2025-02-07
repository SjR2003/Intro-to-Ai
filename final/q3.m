clc; clear; close all;

%% preprocessing
data = load('evaporator.dat'); 

if any(isnan(data), 'all')
    disp('Data contains NaN values. Replacing NaNs with column mean...');
    for i = 1:size(data, 2)
        col = data(:, i);
        col_mean = mean(col(~isnan(col))); 
        col(isnan(col)) = col_mean; 
        data(:, i) = col;
    end
end

data = normalize(data); 

train_ratio = 0.7;
n = size(data, 1);
idx = randperm(n); 
train_data = data(idx(1:round(train_ratio * n)), :);
test_data = data(idx(round(train_ratio * n) + 1:end), :);

X_train = train_data(:, 1:end-1); 
y_train = train_data(:, end); 
X_test = test_data(:, 1:end-1); 
y_test = test_data(:, end); 

%% RBF net
spread = 1.5; 
goal = 0.001; 
max_neurons = 10; 
display_freq = 1; 

net_rbf = newrb(X_train', y_train', goal, spread, max_neurons, display_freq);

y_pred_rbf = sim(net_rbf, X_test')';

rmse_rbf = sqrt(mean((y_pred_rbf - y_test).^2));
disp(['RBF RMSE: ', num2str(rmse_rbf)]);

%% ANFIS net
% ??????? ????? ANFIS
num_mf = 3; 
mf_type = 'gaussmf'; 
epoch_n = 4; 

% ????? ???????? ??????
train_fis_data = [X_train, y_train];

% ????? ???? ??? ?? ?????????? ? ??? ????
val_ratio = 0.3; % ???? ???????? ??????????
n_val = round(val_ratio * size(test_data, 1));
val_data = test_data(1:n_val, :); % ???????? ??????????
test_data_new = test_data(n_val+1:end, :); % ???????? ??? ?????

% ???? FIS ?????
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = num_mf;
opt.InputMembershipFunctionType = mf_type;
fis = genfis(X_train, y_train, opt);

% ????? ANFIS
[fis_trained, trainError, chkError] = anfis(train_fis_data, fis, epoch_n, [], val_data);

% ???????? ?????
y_pred_anfis = evalfis(test_data_new(:, 1:end-1), fis_trained);

% ?????? ???
y_actual = test_data_new(:, end);
mse_anfis = mean((y_actual - y_pred_anfis).^2);
rmse_anfis = sqrt(mse_anfis);

% ????? ?????
fprintf('ANFIS MSE: %.4f\n', mse_anfis);
fprintf('ANFIS RMSE: %.4f\n', rmse_anfis);
