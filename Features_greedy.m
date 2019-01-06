function [besttau, rates, groups, trainrates] = Features_greedy(x, y, nfolds, features, tau)
%This code uses n-fold cross-validation to conduct feature selection to find the best parameter for the greedy heuristic feature selection approach
%procedure: CV first to find the best tau, use the same tau to rerun on the whole data
%return: the best tau, cv accuracy rates of all taus tried, the groups selected, accuracy on the training set during cv
%The same coordinate for different classes are considered as a group.
%
%x: input data
%y: input labels
%folds: number of folds in cv
%features: number of features to select

if (nargin < 2)
	return;
end
l = length(y);
[m,n] = size(x);
if n == l
	x = x';
	[m,n] = size(x);
elseif m ~= l
	fprintf('Dimensionality mismatch\n');
	return;
end

if (nargin < 3)
	nfolds = 10;
end
if (nargin < 4)
	features = 16;
end
if (nargin < 5)
	tau = 1;
end

taus = length(tau);
rates = zeros(taus,1);
trainrates = zeros(taus,1);
grouping = randi(nfolds, l, 1);
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
for i=1:taus
	current_tau = tau(i)
	foldgroups = [];
	for k=1:nfolds
		B = zeros(nFeature, nOutcome);
		k
		train_id = find(grouping ~= k);
		test_id = find(grouping == k);
		x_train = x(train_id,:);
		y_train = y(train_id);
		y_test = y(test_id);
		fn = @(B)func_mlr(B, x_train', y_train);
		G = G0;

		nGroupSelected = 0;
		while(nGroupSelected < min(nGroup - nPreselectedGroup, features))
			tic;
			B = SpaRSA_group(fn, current_tau, B, group, sort(G));
			toc;
			norm_Bi = norm_group(B, group);
			[~, idx_group] = sort(norm_Bi, 'descend');
			idx_selected = find(~ismember(idx_group, G),1);
			group_selected = idx_group(idx_selected);
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
		foldgroups = [foldgroups;G];
		x_train = x(train_id, G);
		x_test = x(test_id, G);
		fn = @(B)func_mlr(B, x_train', y_train);
		Bsub = B(G,:);
		tic;
		[Bsub] = LBFGS_MLE(fn, Bsub);
		output = x_test * Bsub;
		[~,pred] = max(output');
		pred = pred';
		tmprate = sum(pred == y_test) / length(y_test);
		rates(i) = rates(i) + tmprate;
		output = x_train * Bsub;
		[~,pred] = max(output');
		pred = pred';
		tmprate = sum(pred == y_train) / length(y_train);
		trainrates(i) = trainrates(i) + tmprate;
	end
	groups{i} = foldgroups;
end

[~,best] = max(rates);
besttau = tau(best);
