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

%Here the arrays for the possible values for C and sigma are created
C_values=[0.01, 0,03, 0.1, 0.3, 1, 3, 10, 30];
Sig_values=[0.01, 0,03, 0.1, 0.3, 1, 3, 10, 30];

%The matrix for the values of the prediction error is initialised
predictions=zeros(length(C_values),length(Sig_values)); %Maybe better ones(i,j)?

for i=1:length(C_values) %We start filling the prediction error matrix for each combination of C and sigma prediction(i,j) is the prediction error for the i-C and the j-sigma.
    for j=1:length(Sig_values)
        model = svmTrain(X, y, C_values(i), @(x1, x2)gaussianKernel(x1, x2, Sig_values(j))); %Training the model with X and y
        predictor= svmPredict(model, Xval); %Predicting with Xval
        predictions(i,j) = mean(double(predictor ~= yval)); %Comparing to yval to obtain the metric (in this case prediction error)
    end
end

minMatrix = min(predictions(:)); %What (row, column) contains the minimum metric?
[row,col] = find(predictions==minMatrix);

C=C_values(row); %Here we obtain the best values for C and sigma
sigma=Sig_values(col);

% =========================================================================

end
