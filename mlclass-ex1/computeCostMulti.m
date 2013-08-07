function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


s = 0;
i = 1;
j = 0;


while i <= m,
   zwischensumme = 0;
   for j = 1:size(X, 2)
      zwischensumme = (zwischensumme + (X(i, j)) * theta(j)); 
   end 
    s = (s + (zwischensumme - y (i, 1))^2);
    i = i + 1;
end

J = ( s / ( m * 2));





% =========================================================================

end
