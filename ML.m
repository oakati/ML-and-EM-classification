function [test_targets_predicts, param_struct] = ML(train_patterns, train_targets, test_patterns, test_targets, ~)

% Classify using the maximum-likelyhood algorithm
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%	params  		- Unused
%
% Outputs
%	test_targets	- Predicted targets

Uclasses = unique(train_targets);

for i = 1:length(Uclasses),
    indices = find(train_targets == Uclasses(i));
    
    %Estimate mean and covariance
    param_struct(i).mu      = mean(train_patterns(:,indices)');
    param_struct(i).sigma   = cov(train_patterns(:,indices)',1);
    param_struct(i).p       = length(indices)/length(train_targets);
    param_struct(i).w       = 1/length(Uclasses);
    param_struct(i).type    = 'Gaussian';
end

%Classify test patterns
[test_targets_predicts, p_error_given_x] = classify_paramteric(param_struct, test_patterns');

Uclasses = unique(test_targets_predicts);

for i = Uclasses
    indices = (test_targets_predicts == i);
    b = p_error_given_x(:,indices);
    a = max(b);
    param_struct(i).p_error_given_x = (sum(b)-a)./sum(b);
end