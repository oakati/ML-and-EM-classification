function [patterns, train_targets, w, J_W] = MultipleDiscriminantAnalysis(train_patterns, train_targets, c)

%Reshape the data points using multiple descriminant analysis
%Inputs:
%	train_patterns	- Input patterns
%	train_targets	- Input targets
%
%Outputs
%	patterns			- New patterns
%	targets			- New targets
%  w					- Weights vector

[D, ~]	= size(train_patterns);
Utargets	= unique(train_targets);
x			= train_patterns;

if (size(train_patterns,1) < length(Utargets)-1)
   error('Number of classes must be equal to or smaller than input dimension')
end

%Estimate Sw and Sb
m		= mean(x,2);

Sw_i	= zeros(length(Utargets), D, D);
Sb_i	= zeros(length(Utargets), D, D);

for i = 1:length(Utargets)
   indices		= find(train_targets == Utargets(i)); 
   m_i			= mean(x(:,indices),2);

   Sw_i(i,:,:) = (x(:, indices) - m_i*ones(1,length(indices)))*(x(:, indices) - m_i*ones(1,length(indices)))';
   Sb_i(i,:,:) = length(indices)*(m_i - m)*(m_i - m)';
end

Sw		= squeeze(sum(Sw_i));
Sb		= squeeze(sum(Sb_i));

[v, d]= eig(Sb, Sw);
d = diag(d);
[~,index] = sort(abs(d),'descend');
v = v(:,index);
d = d(index);
d = diag(d);

w			= v(:,1:c-1);
patterns = w'*x;

J_W = det(w'*Sb*w)/det(w'*Sw*w);