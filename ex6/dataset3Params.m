function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
if 1
C_tmp     = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma_tmp  = [0.01;0.03;0.1;0.3;1;3;10;30];
val_err   = zeros(length(C_tmp)*length(sigma_tmp),3);

for i=1:length(C_tmp)
  for j=1:length(sigma_tmp)
    model= svmTrain(X, y, C_tmp(i,:), @(x1, x2) gaussianKernel(x1, x2, sigma_tmp(j,:)));
    pred = svmPredict(model, Xval);
    err_tmp = mean(double(pred == yval));
    val_err((i-1)*length(sigma_tmp)+j, :) = [C_tmp(i,:) sigma_tmp(j,:) err_tmp];
  end
end

[M I] = max(val_err);
highest_score_index = I(3);
C = val_err(highest_score_index, 1);
sigma = val_err(highest_score_index, 2);
% =========================================================================
end
end
