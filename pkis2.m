if (exist('min_clusters','var') == 0)
	min_clusters = 2;
end
if (exist('max_clusters','var') == 0)
	max_clusters = 5;
end

load PKIS2_1um;

l = length(targets)
features = 16
repeat = 100
folds = 5
threshold = 20
fold_size = round(l / folds);
auc = zeros(max_clusters, 2);
correct = zeros(max_clusters, 2);
total = zeros(max_clusters,1);
n = size(X,1)
group = cell(n + 1,1);
for i=1:n+1
	group{i} = i; %each group has 1 coordinate, by nature the sparsa code groups the coordinates in one group for all classes together
end
G0 = [n + 1];%do not penalize the bias term
nGroup = length(group);
nPreselectedGroup = length(G0);

for f=1:folds
	f
	test_id = split(fold_size * (f-1) + 1: min(l, fold_size * f));
	train_id = split;
	train_id(fold_size * (f-1) + 1: min(l, fold_size * f)) = [];
	X_train = X(:,train_id)';
	X_test = X(:,test_id)';
	y_s_orig = X_test >= threshold;
	mt = size(X_test,1);
	E = X_train;
	[m,n] = size(E);
	denominator = max(E) - min(E);
	intercept = min(E);
	E = (E - repmat(min(E),[m,1])) ./ repmat(denominator,[m,1]);
	X_test = (X_test - repmat(intercept,[mt,1])) ./ repmat(denominator,[mt,1]);
	E = [E ones(m,1)];
	X_test = [X_test ones(mt,1)];
	n = n + 1;
	for k=min_clusters:max_clusters
		k
		y_s = y_s_orig;
		[idx, C, sumd, D] = kmeans(E, k, 'Replicates', repeat);
		C = C(:,1:end-1);
		C = (C .* repmat(denominator,[k,1])) + repmat(intercept,[k,1]);
		tau = 1e-4;
		nOutcome = k;
		nGroupSelected = 0;
		while (nGroupSelected ~= features)
			tau = tau * 0.1
			B = zeros(n, nOutcome);
			fn = @(B)func_mlr(B, E', idx);
			G = G0;
			nGroupSelected = 0;
			while(nGroupSelected < min(nGroup - nPreselectedGroup, features))
				B = SpaRSA_group(fn, tau, B, group, sort(G));
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
		end
		G = sort(G);
		idx_feature = sort([group{G}]);
		x_train = E(:,G);
		fn = @(B)func_mlr(B, x_train', idx, 0);
		Bsub = B(G,:);
		[Bsub] = LBFGS_MLE(fn, Bsub);
		x_test = X_test(:,G);
		Bx = x_test * Bsub;
		%need to deal with numerical issues
		Bx = Bx';
		[nBx, mBx] = size(Bx);
		[max_val, max_idx] = max(Bx);
		Bx_mod = Bx-repmat(max_val, nBx, 1);

		exp_Bx     = exp(Bx_mod);
		sum_exp_Bx = sum(exp_Bx);
		log_sum_exp_Bx = max_val + log(sum_exp_Bx);
		probs = exp(Bx - repmat(log_sum_exp_Bx, k, 1));

		[~,predcluster] = max(probs);
		G = G(1:end-1);
		approach1 = C(predcluster,:);
		approach1(:,G) = x_test(1:end-1);
		approach2 = probs' * C;
		approach2(:,G) = x_test(1:end-1);
		y_s(:,G) = [];
		approach1(:,G) = [];
		approach2(:,G) = [];
		auctmp = evaluate(approach1', y_s');
		auc(k,1) = auc(k,1) + auctmp;
		correct(k,1) = correct(k,1) + sum(sum((approach1 >= threshold) == y_s));
		auctmp = evaluate(approach2', y_s');
		auc(k,2) = auc(k,2) + auctmp;
		correct(k,2) = correct(k,2) + sum(sum((approach2 >= threshold) == y_s));
		[tmp1,tmp2] = size(y_s);
		total(k) = total(k) + tmp1 * tmp2;
	end
end
auc = auc ./ l;
