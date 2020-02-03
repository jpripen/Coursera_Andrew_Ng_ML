function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%PART 1 OF THE EXERCISE%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Transformation of the y vector into labels
lab=(1:num_labels)';
y_v=[];
for i=1:m
    y_v=[y_v lab==y(i)];
end
y_v=y_v';
y_v=logical(y_v);
%

%Forward propagation of the network. We need to add a column of ones to
%take into account the bias elements for every layer.
a1=[ones(m,1) X];

z2=a1*Theta1';
a2=sigmoid(z2);
a2=[ones(m,1) a2];

z3=a2*Theta2';
a3=sigmoid(z3);

%Removing of the bias columns in theta_i
Theta1_nobias=Theta1(:,2:end);
Theta2_nobias=Theta2(:,2:end);

%Calculation of the cost function
cost_or=sum(sum((1/m)*((-y_v.*log(a3))-(1-y_v).*log(1-a3))));
cost_reg=(lambda/(2*m))*(sum(sum(Theta1_nobias.^2))+sum(sum(Theta2_nobias.^2)));

J=cost_or+cost_reg;

%PART 2 OF THE EXERCISE%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Delta_1=0;
Delta_2=0;

z2_add=[ones(m,1) z2];

for t=1:m
    delta_3=(a3(t,:)-y_v(t,:))';
    delta_2=Theta2'*delta_3.*sigmoidGradient((z2_add(t,:))');
    
    Delta_1=Delta_1+delta_2(2:end)*a1(t,:);
    Delta_2=Delta_2+delta_3*a2(t,:);
end

Theta1_reg=[zeros(size(Theta1,1),1) Theta1_nobias];
Theta2_reg=[zeros(size(Theta2,1),1) Theta2_nobias];

Theta1_grad_reg=(lambda/m)*Theta1_reg;
Theta2_grad_reg=(lambda/m)*Theta2_reg;

Theta1_grad_nor=Delta_1/m;
Theta2_grad_nor=Delta_2/m;

Theta1_grad=Theta1_grad_nor+Theta1_grad_reg;
Theta2_grad=Theta2_grad_nor+Theta2_grad_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
