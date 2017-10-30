function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = length(theta);
tmp = zeros(1,n);

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
