function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
Jv = 0;
ungrad = zeros(size(theta));
grad = zeros(size(theta));
vungrad = zeros(size(theta));
vgrad = zeros(size(theta));
n = length(grad);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


J = (1 / m) * sum(-y(:,1) .* log(sigmoid(X(:, :) * theta)) - (1 - y(:,1)) .* log( 1 - sigmoid(X(:, :) * theta ))) + ((lambda / (2 * m )) * sum(theta(2:n) .^2) );

%Jv = (1 / m) * (-y(:)' * log(sigmoid(X(:, :) * theta)) - (1 - y(:))' * log( 1 - sigmoid(X(:, :) * theta ))) .+ ((lambda / (2 * m )) * sum(theta(2:end) .^2));

%Jv;

# first feature must NOT be regularized!!!

%grad(1) = [  1 / m * sum((sigmoid(X(:, :) * theta) - y(:,1)) .* X(:, 1) ); ];

%for i = 2:n,
% i;
% grad(i) = [  1 / m * (sum((sigmoid(X(:, :) * theta) - y(:,1)) .* X(:, i) ) + (lambda * theta(i)) ); ];
%end

%grad;

%lambda;

%theta(2);

%lambda * theta(2);

%1 / m * (sum((sigmoid(X(:, :) * theta) - y(:,1)) .* X(:, 2) ) + (lambda * theta(2)) );

%1 / m * (sum((sigmoid(X(:, :) * theta) - y(:,1)) .* X(:, 2) ));



grad = (X' * (sigmoid(X(:, :) * theta) - y(:)));

temp = theta;
temp(1) = 0;

grad = grad + (lambda .* temp(:));
grad = grad ./ m;
size(grad);

%vungrad =   (X' * (sigmoid(X(:, :) * theta) - y(:))) ./ m ;

%vungrad;

%for i = 1:n,
% ungrad(i) = [  1 / m * sum((sigmoid(X(:, :) * theta) - y(:,1)) .* X(:, i) ) ; ];
%end

%ungrad;


% =============================================================

end
