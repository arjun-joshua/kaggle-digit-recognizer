function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a THREE layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size1, hidden_layer_size2, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2 AND Theta3, the weight matrices
% for our THREE layer neural network
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

nn_params(1:hidden_layer1_size * (input_layer_size + 1)) = [];%cut out parameters allotted in previous step

Theta2 = reshape(nn_params(1:hidden_layer2_size * (hidden_layer1_size + 1)), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));
             
nn_params(1:hidden_layer2_size * (hidden_layer1_size + 1)) = [];%cut out parameters allotted in previous step

Theta3 = reshape(nn_params, ...
                 num_labels, (hidden_layer2_size + 1));             

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
% % J = 0; %COMMENTED OUT TO AVOID PREALLOCATION WARNING BY MATLAB
% % Theta1_grad = zeros(size(Theta1));
% % Theta2_grad = zeros(size(Theta2));
% % Theta3_grad = zeros(size(Theta3));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J.
a1 = [ones(1,m) ; X'];%input unit (matrix containing all training data) with bias

z2 = Theta1 * a1;    
a2 = [ones(1,m) ; sigmoid(z2)];%hidden layer 1 (matrix over all training data) with bias
    
z3 = Theta2 * a2;    
a3 = [ones(1,m) ; sigmoid(z3)];%hidden layer 2 (matrix over all training data) with bias

a4 = sigmoid(Theta3 * a3);%output unit (matrix containing output vectors for all training data)

Y = zeros(num_labels,m);
Y(sub2ind(size(Y),y',1:m)) = 1;%output labels made into a vector and concatenated into a matrix

J = -(1/m) * sum(sum( Y.*log(a4) + (1 - Y).*log(1 - a4) ));
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad AND Theta3_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 AND Theta3 in Theta1_grad and
%         Theta2_grad AND Theta3_grad, respectively. After implementing Part 2, you can check
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
delta4 = a4 - Y; %size = 10 X 5000

%25 X 5000        25 X 10      10 X 5000     25 X 5000
delta3 = ( (Theta3(:,2:end))' * delta4 ) .* sigmoidGradient(z3);

%25 X 5000        25 X 25      25 X 5000     25 X 5000
delta2 = ( (Theta2(:,2:end))' * delta3 ) .* sigmoidGradient(z2);

% 10 X 26          10 X 5000   5000 X 26
Theta3_grad = (1/m) * delta4 * (a3)';

% 25 X 26          25 X 5000   5000 X 26
Theta2_grad = (1/m) * delta3 * (a2)';

%  25 X 785        25 X 5000   5000 X 785
Theta1_grad = (1/m) * delta2 * (a1)';

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad AND Theta3_grad from Part 2.
%
J = J + (lambda/(2*m)) * ( Theta1(size(Theta1,1)+1:end) * Theta1(size(Theta1,1)+1:end)' +...
                           Theta2(size(Theta2,1)+1:end) * Theta2(size(Theta2,1)+1:end)' +...
                           Theta3(size(Theta3,1)+1:end) * Theta3(size(Theta3,1)+1:end)' );

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);                           
    
Theta3_grad(:,2:end) = Theta3_grad(:,2:end) + (lambda/m) * Theta3(:,2:end);                           


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
