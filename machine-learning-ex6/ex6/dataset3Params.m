function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
min_error = 999;
min_c = 0;
min_sigma = 0;

%C_List = [0.01, 0.03, 0.1, 0.3, 1, 3 , 10, 30];
%sigma_List = [0.01, 0.03, 0.1, 0.3, 1, 3 , 10, 30];
C_List = [0.1,0.13,0.16,0.19,0.2,0.22,0.25];
sigma_List = [0.06,0.063,0.066,0.07,0.073,0.076,0.08];
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

for i=1:length(C_List)
    for j=1:length(sigma_List)
        model= svmTrain(X, y, C_List(i), @(x1, x2) gaussianKernel(x1, x2, sigma_List(j)));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        if err<min_error
            min_error = err;
            min_c = C_List(i);
            min_sigma = sigma_List(j);
        end
    end
end
C = min_c;
sigma = min_sigma;
end
