function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
tmp = zeros(1,n);
J_history = zeros(num_iters, 1);

%for iter = 1:num_iters
%    tmp1 = 0;
%    tmp2 = 0;
%    for i = 1:m
%        tmp1 = tmp1+(theta(1)*X(i,1)+theta(2)*X(i,2)-y(i))*X(i,1);
%        tmp2 = tmp2+(theta(1)*X(i,1)+theta(2)*X(i,2)-y(i))*X(i,2);
%    end
%    theta(1) = theta(1) - alpha/m*tmp1;
%    theta(2) = theta(2) - alpha/m*tmp2;
%
%    % Save the cost J in every iteration    
%    J_history(iter) = computeCost(X, y, theta);
%    
%    %fprintf('J thera now is %f\n',J_history(iter));
%end


for iter = 1:num_iters
    for i = 1:n
        tmp(i) = sum((X*theta-y).*X(:,i));
    end
 
    for i = 1:n
        theta(i) = theta(i) - alpha*tmp(i)/m;
    end
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    %fprintf('J thera now is %f\n',J_history(iter));
end

end
