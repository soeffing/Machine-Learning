function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

# tests

#X = [ones(m, 1) X];
#theta = [1; 1; 1];

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


#funktion = theta(1) .+ (X .* theta(2));

#size(funktion);
#size(y);

# part1 =  ((theta(1) .* X(:,1) ) + (X(:,2) .* theta(2)) - y).^2;

part1 = ((X * theta )- y).^2;

sum(part1);

part2 = sum(part1) / (2*m);

#size(test)

part3 = (lambda / (m * 2)) * sum(theta(2:end).^2); 

J = part2 + part3;


#s = 0;
#i = 1;

#while i <= m,
# s = (s + (((X(i, 1) * theta(1)) + (X(i, 2) * theta(2))) - y (i, 1))^2);
# i = i + 1;
#end



#J = ( s / ( m * 2));

# grad

#grad(1) = sum ( ((X * theta )- y) .* X(:,1) );
#grad(1) = grad(1) / m;

grad1 = ((X * theta )- y)' * X;
grad(:) = grad1' / m;

grad(2:end) = grad(2:end) + ( (lambda / m) * theta(2:end) );

#grad = sum ( ((X * theta )- y)' * X);
#grad(1) = grad(1) / m;
#grad(2:end) = (grad(2:end) / m) + ( (lambda / m) * theta(2:end) );








% =========================================================================

grad = grad(:);

end
