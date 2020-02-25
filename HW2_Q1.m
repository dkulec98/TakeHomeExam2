%% ================== Generate and Plot Training Set ================== %%
clear all; close all; clc;

n = 2;      % number of feature dimensions
N_10 = 10;  % number of iid samples
N_100 = 100;
N_1000 = 1000;

% parallel distributions
mu(:,1) = [-2;0]; Sigma(:,:,1) = [1 -0.9;-0.9 2];
mu(:,2) = [2;0]; Sigma(:,:,2) = [2 0.9;0.9 1]; 
%mu(:,1) = [3;0]; Sigma(:,:,1) = [5 0.1;0.1 .5]; 
%mu(:,2) = [0;0]; Sigma(:,:,2) = [.5 0.1;0.1 5];

% Class priors for class 0 and 1 respectively
p = [0.5,0.5]; 

% Generating true class labels
label_10 = (rand(1,N_10) >= p(1))';
Nc_10 = [length(find(label_10==0)),length(find(label_10==1))];

% Draw samples from each class pdf
x = zeros(N_10,n); 
for L = 0:1
    x(label_10==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_10(L+1));
end

%Plot samples with true class labels
figure(1);
plot(x(label_10==0,1),x(label_10==0,2),'o',x(label_10==1,1),x(label_10==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels (N=10)');
xlabel('x_1'); ylabel('x_2'); hold on;

%% ======================== Logistic Regression ======================= %%
% Initialize fitting parameters
x = [ones(N_10, 1) x];
initial_theta = zeros(n+1, 1);
label_10=double(label_10);

% Compute gradient descent to get theta values
[theta, cost] = gradient_descent(x,N_10,label_10,initial_theta,1,10);
[theta2, cost2] = fminsearch(@(t)(cost_func(t, x, label_10, N_10)), initial_theta);

% Choose points to draw boundary line
plot_x1 = [min(x(:,2))-2,  max(x(:,2))+2];                      
plot_x2(1,:) = (-1./theta(3)).*(theta(2).*plot_x1 + theta(1));  
plot_x2(2,:) = (-1./theta2(3)).*(theta2(2).*plot_x1 + theta2(1)); % fminsearch

% Plot decision boundary
plot(plot_x1, plot_x2(1,:), plot_x1, plot_x2(2,:));  
axis([plot_x1(1), plot_x1(2), min(x(:,3))-2, max(x(:,3))+2]);
legend('Class 0', 'Class 1', ' Classifier (from scratch)', 'Classifier (fminsearch)');

% Plot cost function
figure(2); plot(1-cost);
title('Calculated Cost (N=10)');
xlabel('Iteration number'); ylabel('Cost');

%% ================= Training and Regression for N=100 ================= %%

label_100 = (rand(1,N_100) >= p(1))';
Nc_100 = [length(find(label_100==0)),length(find(label_100==1))];

% Draw samples from each class pdf
x = zeros(N_100,n); 
for L = 0:1
    x(label_100==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_100(L+1));
end

figure(3);
plot(x(label_100==0,1),x(label_100==0,2),'o',x(label_100==1,1),x(label_100==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels (N=100)');
xlabel('x_1'); ylabel('x_2'); hold on;

% Initialize fitting parameters
x = [ones(N_100, 1) x];
initial_theta = zeros(n+1, 1);
label_100=double(label_100);

% Compute gradient descent to get theta values
[theta3, cost3] = gradient_descent(x,N_100,label_100,initial_theta,1,100);
[theta4, cost4] = fminsearch(@(t)(cost_func(t, x, label_100, N_100)), initial_theta);

% Choose points to draw boundary line
plot_x3 = [min(x(:,2))-2,  max(x(:,2))+2];                      
plot_x4(1,:) = (-1./theta3(3)).*(theta3(2).*plot_x3 + theta3(1));  
plot_x4(2,:) = (-1./theta4(3)).*(theta4(2).*plot_x3 + theta4(1)); % fminsearch

% Plot decision boundary
plot(plot_x3, plot_x4(1,:), plot_x3, plot_x4(2,:));  
axis([plot_x3(1), plot_x3(2), min(x(:,3))-2, max(x(:,3))+2]);
legend('Class 0', 'Class 1', ' Classifier (from scratch)', 'Classifier (fminsearch)');

% Plot cost function
figure(4); plot(1-cost3);
title('Calculated Cost (N=100)');
xlabel('Iteration number'); ylabel('Cost');

%% ================= Training and Regression for N=1000 ================= %%

label_1000 = (rand(1,N_1000) >= p(1))';
Nc_1000 = [length(find(label_1000==0)),length(find(label_1000==1))];

x = zeros(N_1000,n); 
for L = 0:1
    x(label_1000==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_1000(L+1));
end

figure(5);
plot(x(label_1000==0,1),x(label_1000==0,2),'o',x(label_1000==1,1),x(label_1000==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels (N=1000)');
xlabel('x_1'); ylabel('x_2'); hold on;

% Initialize fitting parameters
x = [ones(N_1000, 1) x];
initial_theta = zeros(n+1, 1);
label_1000=double(label_1000);

% Compute gradient descent to get theta values
[theta5, cost5] = gradient_descent(x,N_1000,label_1000,initial_theta,1,1000);
[theta6, cost6] = fminsearch(@(t)(cost_func(t, x, label_1000, N_1000)), initial_theta);

% Choose points to draw boundary line
plot_x5 = [min(x(:,2))-2,  max(x(:,2))+2];                      
plot_x6(1,:) = (-1./theta5(3)).*(theta5(2).*plot_x5 + theta5(1));  
plot_x6(2,:) = (-1./theta6(3)).*(theta6(2).*plot_x5 + theta6(1)); % fminsearch

% Plot decision boundary
plot(plot_x5, plot_x6(1,:), plot_x5, plot_x6(2,:));  
axis([plot_x5(1), plot_x5(2), min(x(:,3))-2, max(x(:,3))+2]);
legend('Class 0', 'Class 1', ' Classifier (from scratch)', 'Classifier (fminsearch)');
    
% Plot cost function
figure(6); plot(1-cost5);
title('Calculated Cost (N=1000)');
xlabel('Iteration number'); ylabel('Cost');
%% ====================== Generate Test Data Set ====================== %%
N_test = 10000;

% Generating true class labels
label_test = (rand(1,N_test) >= p(1))';
Nc_test = [length(find(label_test==0)),length(find(label_test==1))];

% Draw samples from each class pdf
x_test = zeros(N_test,n); 
for L = 0:1
    x_test(label_test==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_test(L+1));
end

%% ========================= Test Classifier ========================== %%
% Coefficients for decision boundary line equation
coeff_10(1,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(1,1), plot_x2(1,2)], 1);
coeff_10(2,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(2,1), plot_x2(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on
for i = 1:2
    if coeff_10(i,1) >= 0
        decision_10(:,i) = (coeff_10(i,1).*x_test(:,1) + coeff_10(i,2)) < x_test(:,2);
    else
        decision_10(:,i) = (coeff_10(i,1).*x_test(:,1) + coeff_10(i,2)) > x_test(:,2);
    end
end

fprintf('For N=10 Training Samples:\n');
error1 = plot_test_data(decision_10(:,1), label_test, Nc_test, p, 7, x_test, plot_x1, plot_x2(1,:));
title('Test Data Classification (from scratch) (N=10)');
xlabel('x_1'); ylabel('x_2');
fprintf('Total error (classifier from scratch): %.2f%%\n',100-error1);

error2 = plot_test_data(decision_10(:,2), label_test, Nc_test, p, 8, x_test, plot_x1, plot_x2(2,:));
title('Test Data Classification (using fminsearch) (N=10)');
xlabel('x_1'); ylabel('x_2');
fprintf('Total error (classifier using fminsearch): %.2f%%\n',100-error2);


coeff_100(1,:) = polyfit([plot_x3(1), plot_x3(2)], [plot_x4(1,1), plot_x4(1,2)], 1);
coeff_100(2,:) = polyfit([plot_x3(1), plot_x3(2)], [plot_x4(2,1), plot_x4(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on
for i = 1:2
    if coeff_100(i,1) >= 0
        decision_100(:,i) = (coeff_100(i,1).*x_test(:,1) + coeff_100(i,2)) < x_test(:,2);
    else
        decision_100(:,i) = (coeff_100(i,1).*x_test(:,1) + coeff_100(i,2)) > x_test(:,2);
    end
end

fprintf('For N=100 Training Samples:\n');
error3 = plot_test_data(decision_100(:,1), label_test, Nc_test, p, 9, x_test, plot_x3, plot_x4(1,:));
title('Test Data Classification (from scratch) (N=100)');
xlabel('x_1'); ylabel('x_2');
fprintf('Total error (classifier from scratch): %.2f%%\n',100-error3);

error4 = plot_test_data(decision_100(:,2), label_test, Nc_test, p, 10, x_test, plot_x3, plot_x4(2,:));
title('Test Data Classification (using fminsearch) (N=100)');
xlabel('x_1'); ylabel('x_2');
fprintf('Total error (classifier using fminsearch): %.2f%%\n',100-error4);

coeff_1000(1,:) = polyfit([plot_x5(1), plot_x5(2)], [plot_x6(1,1), plot_x6(1,2)], 1);
coeff_1000(2,:) = polyfit([plot_x5(1), plot_x5(2)], [plot_x6(2,1), plot_x6(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on
for i = 1:2
    if coeff_1000(i,1) >= 0
        decision_1000(:,i) = (coeff_1000(i,1).*x_test(:,1) + coeff_1000(i,2)) < x_test(:,2);
    else
        decision_1000(:,i) = (coeff_1000(i,1).*x_test(:,1) + coeff_1000(i,2)) > x_test(:,2);
    end
end

fprintf('For N=1000 Training Samples:\n');
error5 = plot_test_data(decision_1000(:,1), label_test, Nc_test, p, 11, x_test, plot_x1, plot_x2(1,:));
title('Test Data Classification (from scratch) (N=1000)');
xlabel('x_1'); ylabel('x_2');
fprintf('Total error (classifier from scratch): %.2f%%\n',100-error5);

error6 = plot_test_data(decision_1000(:,2), label_test, Nc_test, p, 12, x_test, plot_x1, plot_x2(2,:));
title('Test Data Classification (using fminsearch) (N=1000)');
xlabel('x_1'); ylabel('x_2');
fprintf('Total error (classifier using fminsearch): %.2f%%\n',100-error6);


%% ============================ Functions ============================= %%
function [theta, cost] = gradient_descent(x, N, label, theta, alpha, num_iters)
    cost = zeros(num_iters, 1);
    for i = 1:num_iters % while norm(cost_gradient) > threshold
        h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function   
        cost(i) = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
        cost_gradient = (1/N)*(x' * (h - label));
        theta = theta - (alpha.*cost_gradient); % Update theta
    end
end
   
function cost = cost_func(theta, x, label,N)
    h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function
    cost = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
end

function error = plot_test_data(decision, label, Nc, p, fig, x, plot_x1, plot_x2)
    ind00 = find(decision==0 & label==0); % true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % false negative
    ind11 = find(decision==1 & label==1); % true positive
    error = (p10*p(1) + p01*p(2))*100;

    % Plot decisions and decision boundary
    figure(fig);
    plot(x(ind00,1),x(ind00,2),'or'); hold on,
    plot(x(ind10,1),x(ind10,2),'og'); hold on,
    plot(x(ind01,1),x(ind01,2),'+g'); hold on,
    plot(x(ind11,1),x(ind11,2),'+r'); hold on,
    plot(plot_x1, plot_x2);
    axis([plot_x1(1), plot_x1(2), min(x(:,2))-2, max(x(:,2))+2])
    legend('Class 0 Wrong Decisions','Class 0 Correct Decisions','Class 1 Correct Decisions','Class 1 Wrong Decisions','Classifier');
end