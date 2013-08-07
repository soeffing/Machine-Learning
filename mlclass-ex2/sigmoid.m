function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


%exp = e.^(-z);

%summe = 1 .+ exp;

%g = 1 ./ summe;

% just testing and taking the sigmoid function form ex3

g = 1.0 ./ (1.0 + exp(-z));
 

% =============================================================

end
