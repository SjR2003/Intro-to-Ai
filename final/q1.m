% Define the grid for x and y
[x1, x2] = meshgrid(-2:0.1:2, -2:0.1:2);

% Define the target points t1 and t2
t1 = [1, 1];  % Target point 1
t2 = [-1, -1]; % Target point 2

% Compute the distances and the function values
distance1 = (x1 - t1(1)).^2 + (x2 - t1(2)).^2;
distance2 = (x1 - t2(1)).^2 + (x2 - t2(2)).^2;
y = -exp(-distance1) - exp(-distance2) + 1;

% Plot the surface
figure;
surf(x1, x2, y);
xlabel('x1');
ylabel('x2');
zlabel('y');
title('Plot of the Function');
shading interp;
colorbar;