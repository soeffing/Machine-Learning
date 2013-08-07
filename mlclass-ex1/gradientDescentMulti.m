function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
j = 0;  % für die zwischensumme = dritte for schleife für funktion
i = 0;  % für's theta update = zweite schleife for subtracting y and multiplying with Xij
h = 0;  % first loop for assigning theta simultaneously


% for h = 1:size(theta, 1)

%  zweitezwischensumme = 0;
  
  for i = 1:size(X,2)
    zwischensumme = 0; 
    for j = 1:size(X, 2)
      zwischensumme = ( zwischensumme + (X(:, j)) * theta(j));
     end 
    zweitezwischensumme = (zwischensumme - y(:, 1)) .* X(:, i);
    s(i, 1) = [sum(zweitezwischensumme)];

    if i==size(X,2),

       theta = theta .- (alpha * 1/m * s);
      
    %   for h = 1:size(theta, 1)
    %     theta(h, 1) = [theta(h, 1) - (alpha * (1 / m) * s(h, 1)) ];
    %   end
    end
  end

size(s);
size(theta);

J = computeCostMulti(X, y, theta);


% save J1.txt iter -ascii;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
