clc;
clear;

%% ANFIS generator
n_samples = 5000;
x = -1 + 2 * rand(n_samples, 1); 
y = -1 + 2 * rand(n_samples, 1);

centers = [-2, 0, 2]; 
sigma = 0.5;
gauss_func = @(x, c, s) exp(-((x - c).^2) / (2 * s^2));

mf_x = zeros(n_samples, length(centers));
mf_y = zeros(n_samples, length(centers));
for i = 1:length(centers)
    mf_x(:, i) = gauss_func(x, centers(i), sigma);
    mf_y(:, i) = gauss_func(y, centers(i), sigma);
end

z = zeros(n_samples, 1);
for i = 1:length(centers)
    for j = 1:length(centers)
        rule_output = mf_x(:, i) .* mf_y(:, j); 
        z = z + rule_output .* (centers(i) + centers(j)); 
    end
end

figure;
scatter3(x, y, z, '.');
title('Generator ANFIS');
xlabel('x');
ylabel('y');
zlabel('z')

%% MLP
hidden_layer_size = 10;
inputs = [x, y]';
targets = z';

net = feedforwardnet(hidden_layer_size);
net = train(net, inputs, targets);

z_pred = net(inputs);

figure;
scatter3(x, y, z_pred, '.');
title('MLP');
xlabel('x');
ylabel('y');
zlabel('z');

anfis_params = length(centers)^2 + length(centers) * 2; 
mlp_params = (2 * hidden_layer_size) + hidden_layer_size + hidden_layer_size + 1; 

disp(['ANFIS: ', num2str(anfis_params)]);
disp(['MLP: ', num2str(mlp_params)]);

%% new nfis
n_samples = 1000;
x_new = -2 + 4 * rand(n_samples, 1);
y_new = -2 + 4 * rand(n_samples, 1);
inputs_new = [x_new, y_new]';
z_new = net(inputs_new);

centers_new = linspace(-2, 2, 3); 
sigma_new = 0.5; 
gauss_func = @(x, c, s) exp(-((x - c).^2) / (2 * s^2));

mf_x_new = zeros(n_samples, length(centers_new));
mf_y_new = zeros(n_samples, length(centers_new));
for i = 1:length(centers_new)
    mf_x_new(:, i) = gauss_func(x_new, centers_new(i), sigma_new);
    mf_y_new(:, i) = gauss_func(y_new, centers_new(i), sigma_new);
end

z_anfis_new = zeros(n_samples, 1);
for i = 1:length(centers_new)
    for j = 1:length(centers_new)
        rule_output_new = mf_x_new(:, i) .* mf_y_new(:, j); 
        z_anfis_new = z_anfis_new + rule_output_new .* (centers_new(i) + centers_new(j)); 
    end
end

figure;
scatter3(x_new, y_new, z_new, '.');
title('MLP Outputs');
xlabel('x');
ylabel('y');
zlabel('z');

figure;
scatter3(x_new, y_new, z_anfis_new, '.');
title('Improved ANFIS Outputs');
xlabel('x');
ylabel('y');
zlabel('z');

anfis_params_new = length(centers_new)^2 + length(centers_new) * 2;
disp(['ANFIS params: ', num2str(anfis_params_new)]);

%% Step 4: Validation Metrics
% Compute Errors
errors = z_new - z_anfis_new;

% Compute Evaluation Metrics
MSE = mean(errors.^2);  % Mean Squared Error
RMSE = sqrt(MSE);       % Root Mean Squared Error
MAE = mean(abs(errors));% Mean Absolute Error
R2 = 1 - sum(errors.^2) / sum((z_new - mean(z_new)).^2); % R-squared

% Display Metrics
disp(['MSE: ', num2str(MSE)]);
disp(['RMSE: ', num2str(RMSE)]);
disp(['MAE: ', num2str(MAE)]);
disp(['R^2: ', num2str(R2)]);



