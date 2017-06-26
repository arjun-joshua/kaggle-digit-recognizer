function [error_train, error_val, Theta1, Theta2, Theta3] = ...
    learningCurve(X, y, Xval, yVal, ...
    input_layer_size, hidden_layer1_size, hidden_layer2_size, num_labels, lambda, nIter, ...
    initial_Theta1, initial_Theta2, initial_Theta3)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
M = 1;%12;%m = size(X, 1);

% You need to return these values correctly
error_train = zeros(M, 1);
error_val   = zeros(M, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
mVal = size(Xval,1);

if ~exist('initial_Theta1', 'var') || ~exist('initial_Theta2', 'var') || ~exist('initial_Theta3', 'var') ...
        || isempty(initial_Theta1) || isempty(initial_Theta2)|| isempty(initial_Theta3)
    fprintf('\nInitializing Neural Network Parameters ...\n')
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
    initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
    initial_Theta3 = randInitializeWeights(hidden_layer2_size, num_labels);
end

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

options = optimset('MaxIter', nIter);

for i = 1:M
    fprintf('\nTraining iteration number: %d\n', i)
    Xsel = X( 1 : size(X,1)*i/M , :);
    ySel = y( 1 : size(X,1)*i/M );
    XvalSel = Xval( 1 : size(Xval,1)*i/M , :); 
    yValSel = yVal( 1 : size(Xval,1)*i/M );
    
    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, Xsel, ySel, lambda);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    % Obtain Theta1 and Theta2 AND Theta3 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

    nn_params(1:hidden_layer1_size * (input_layer_size + 1)) = [];%cut out parameters allotted in previous step

    Theta2 = reshape(nn_params(1:hidden_layer2_size * (hidden_layer1_size + 1)), ...
                    hidden_layer2_size, (hidden_layer1_size + 1));
             
    nn_params(1:hidden_layer2_size * (hidden_layer1_size + 1)) = [];%cut out parameters allotted in previous step             

    Theta3 = reshape(nn_params, ...
                    num_labels, (hidden_layer2_size + 1));

%    error_train(i) = linearRegCostFunction( X(1:i,:), y(1:i), theta, 0 );
    pred = predict(Theta1, Theta2, Theta3, Xsel);

    error_train(i) = 100 - mean(double(pred == ySel)) * 100;

    pred = predict(Theta1, Theta2, Theta3, XvalSel);
    
%    error_val(i) = linearRegCostFunction( Xval, yval, theta, 0);
    error_val(i) = 100 - mean(double(pred == yValSel)) * 100;    
end




% -------------------------------------------------------------

% =========================================================================

end
