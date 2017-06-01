function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X * theta;

j0 = h - y;
j0sq = arrayfun (  @(x) x * x, j0);
j1 = 1/ (2*m ) * sum(j0sq);
g1 = 1/m * transpose(X) * ( h - y );

theta2 = theta;
theta2(1) = [];

theta2sq = arrayfun (  @(x) x * x, theta2);
addit = ( lambda  / (2 * m) ) * sum(theta2sq);
J = j1 + addit;

extras = (lambda / m ) * theta;
grad = g1 + extras;
grad(1) = grad(1) - extras(1);



% =========================================================================

grad = grad(:);

end
