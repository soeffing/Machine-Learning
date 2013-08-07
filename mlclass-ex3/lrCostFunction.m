function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
%X = [ones(m,1) X]; % add bias to X
%theta = zeros(size(X,2), 1);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

n = size(X, 2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%




% unregularized 

J = (1 / m) * sum(-y(:,1) .* log(sigmoid(X(:, :) * theta)) - (1 - y(:,1)) .* log( 1 - sigmoid(X(:, :) * theta )));


% unregularized gradient


grad = (X' * (sigmoid(X(:, :) * theta) - y(:))) ./ m;


% regularized cost function

J = (1 / m) * sum(-y(:,1) .* log(sigmoid(X(:, :) * theta)) - (1 - y(:,1)) .* log( 1 - sigmoid(X(:, :) * theta ))) + ((lambda / (2 * m )) * sum(theta(2:end) .^2) );


%regularized gradient

temp = theta;
temp(1) = 0;

grad = (X' * (sigmoid(X(:, :) * theta) - y(:))) + (lambda .* temp(:));

grad = grad ./ m;



% =============================================================

end
