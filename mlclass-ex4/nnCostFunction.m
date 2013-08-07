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

size(Theta1);

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

size(Theta2);

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

# add ones to X

X = [ones(m, 1) X];


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


# converting y into a k-dimensional vector (K = number of classes = num_labels)

yvector = zeros(size(y), num_labels);

for i=1:m
 index = y(i);
 yvector(i, index) = 1;
end




z2 = X * Theta1';

size(z2);

a2 = sigmoid(z2(:,:));

size(a2);

a2 = [ ones(m,1) a2 ];

z3 = a2 * Theta2';

a3 = sigmoid(z3(:,:));

size(a3);

# Cost function

sumk = sum( ( -yvector(:,:) .* log(a3(:,:)) ) - ((1 - yvector(:,:)) .* log( 1 - a3(:,:)  ) ), 2 );
size(sumk);

summ = sum(sumk);

# unregularized cost function result

J = (1 / m) * summ;

# cost function without K - sum
# Remarks: substitute "sigmoid(X(:,:) + theta)) with a3

# J = (1 / m) * sum(-y(:,1) .* log(sigmoid(X(:, :) * theta)) - (1 - y(:,1)) .* log( 1 - sigmoid(X(:, :) * theta )));


# Important note from instructions

# Note that you should not be regularizing the terms that correspond to
#the bias. For the matrices Theta1 and Theta2, this corresponds to the first
# column of each matrix.


#implement regularization for cost function

sumtheta1k = sum(Theta1(:,2:end) .^ 2, 2);
size(sumtheta1k);

sumtheta1j = sum(sumtheta1k);

size(sumtheta1j);

sumtheta2k = sum(Theta2(:, 2:end) .^ 2, 2);
size(sumtheta2k);

sumtheta2j = sum(sumtheta2k);

size(sumtheta2j);

reg = (lambda / (2*m)) * ( sumtheta1j + sumtheta2j);

J = J + reg;


% regularized cost function copied from lrCostFunction mlclass-ex3

#J = (1 / m) * sum(-y(:,1) .* log(sigmoid(X(:, :) * theta)) - (1 - y(:,1)) .* log( 1 - sigmoid(X(:, :) * theta ))) + ((lambda / (2 * m )) * sum(theta(2:end) .^2) );


# Backward Propagation

# initiliaze Deltas!

#required dimension not very clear - updateing delta after each loop
#Del_1 = zeros(size(Theta1, 1), (size(Theta1, 2) - 1));
#Del_2 = zeros(size(Theta2, 1), (size(Theta2, 2) - 1));

Del_1 = zeros(size(Theta1));
Del_2 = zeros(size(Theta2));

size(Del_1);
size(Del_2);

for t = 1:m

 a_1 = X(t,:);
 a_1 = a_1';  # make it to a column vector

 z_2 = a_1' * Theta1';

 size(z_2);

 a_2 = sigmoid(z_2);
  
 size(a_2);

 a_2 = [ones(1, 1) a_2];

 size(a_2);

 z_3 = a_2 * Theta2';

 a_3 = sigmoid(z_3);

 a_3;
  
# step 2 

# preparing yvector -> k-dimensional

 yvector = zeros(1, num_labels);

 index = y(t);
 yvector(1, index) = 1;

 delta_3 = (a_3 - yvector);

 size(delta_3);

 delta_3 = delta_3';

# step 3 hidden layer

 delta_2 = Theta2' * delta_3;

 size(delta_2);

 grad = sigmoidGradient(z_2);
 delta_2 = delta_2(2:end) .* grad';
 
  size(delta_2);

# step 4 aggregating delta

#initialize Deltas before for loop!

 size(delta_3);
 size(a_2);

 Del_1 = Del_1 + (delta_2 * a_1');
 Del_2 = Del_2 + (delta_3 * a_2);


end

# Step 5 of backward propagation

Theta1_grad = ( 1 / m) * Del_1;
Theta2_grad = ( 1 / m) * Del_2;

# Regularization

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda / m ) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda / m ) * Theta2(:, 2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
