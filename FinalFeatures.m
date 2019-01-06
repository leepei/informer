function [groups] = Features_CV_L2regularized(x, y, tau, features)
%This code runs for a set of given parameters (tau, features) the greedy heuristic feature selection approach to select features
%return: the features selected
%
%x: input data
%y: input labels
%tau: the parameter for trading off between the loss and the regularizer
%features: number of features to select


groups = [];
l = length(y);
[m,n] = size(x);
if n == l
	x = x';
	[m,n] = size(x);
elseif m ~= l
	fprintf('Dimensionality mismatch\n');
	return;
end

nFeature = n;
nOutcome = length(unique(y));

G0 = [n];%do not penalize the bias term
group = cell(n,1);
for i=1:n
	group{i} = i; %each group has 1 coordinate, by nature the sparsa code groups the coordinates in one group for all classes together
end

nGroup = length(group);
nPreselectedGroup = length(G0);
groups = {};

current_tau = tau;
B = zeros(nFeature, nOutcome);
x_train = x;
y_train = y;
fn = @(B)func_mlr(B, x_train', y_train);
G = G0;
nGroupSelected = 0;
while(nGroupSelected < min(nGroup - nPreselectedGroup, features))
	B = SpaRSA_group(fn, current_tau, B, group, sort(G));
	norm_Bi = norm_group(B, group);
	[~, idx_group] = sort(norm_Bi, 'descend');
	idx_selected = find(~ismember(idx_group, G),1);
	group_selected = idx_group(idx_selected)
	if norm_Bi(group_selected) > 0
		G = [G group_selected];
		nGroupSelected = nGroupSelected + 1;
		if length(G) == sum(norm_Bi>0)
			break;
		end
	else
		break;
	end
end
G = sort(G);
groups = [groups;G];
