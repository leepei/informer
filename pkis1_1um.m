min_clusters = 3;
max_clusters = 22;
load PKIS1_1um;
load scaff;
[n,l] = size(X);
loostart = 1;
looend = l;
features = 16;
repeat = 100;
folds = 5;
allY = allY';
fold_size = round((l-1) / folds);
ef10 = zeros(max_clusters,1);
rocauc = zeros(max_clusters, 1);
fasr = zeros(max_clusters,1);
counter = 1;
group = cell(n + 1,1);
for i=1:n+1
	group{i} = i;
end
G0 = [n + 1];
nGroup = length(group);
nPreselectedGroup = length(G0);
prediction_ef10 = {};
informer_ef10 = {};
prediction_rocauc = {};
informer_rocauc = {};
prediction_fasr10 = {};
informer_fasr10 = {};

clustersize_ef10 = [];
clustersize_auc = [];
clustersize_fasr10 = [];

for i=loostart:looend
	ef10 = zeros(max_clusters,1);
	rocauc = zeros(max_clusters, 1);
	fasr = zeros(max_clusters, 1);
	if (sum(allY(i,:)) == 0)
		continue;
	end
	split = [1:l];
	split(i) = [];
	for f=1:folds
		test_id = split(fold_size * (f-1) + 1: min(l-1, fold_size * f));
		train_id = split;
		train_id(fold_size * (f-1) + 1: min(l-1, fold_size * f)) = [];
		X_train = X(:,train_id)';
		X_test = X(:,test_id)';
		mt = size(X_test,1);
		y_s_orig = allY(test_id,:);
		E = X_train;
		[m,n] = size(E);
		denominator = max(E) - min(E);
		intercept = min(E);
		E = (E - repmat(min(E),[m,1])) ./ repmat(denominator,[m,1]);
		X_test = (X_test - repmat(intercept,[mt,1])) ./ repmat(denominator,[mt,1]);
		E = [E ones(m,1)];
		mt = size(X_test,1);
		X_test = [X_test ones(mt,1)];
		n = n + 1;
		for k=min_clusters:max_clusters
			y_s = y_s_orig;
			[idx, C, sumd, D] = kmeans(E, k, 'Replicates', repeat);
			C = C(:,1:end-1);
			C = (C .* repmat(denominator,[k,1])) + repmat(intercept,[k,1]);
			tau = 1e-4;
			nOutcome = k;
			nGroupSelected = 0;
			while (nGroupSelected ~= features)
				tau = tau * 0.1;
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

			approach = probs' * C;
			G = G(1:end-1);
			approach(:,G) = x_test(:,1:end-1);
			[ef10tmp,rocauctmp] = evaluate(approach', y_s');
			fasr10tmp = fasr10(approach', y_s', scaffid);
			ef10(k) = ef10(k)+sum(ef10tmp);
			rocauc(k) = rocauc(k) + sum(rocauctmp);
			fasr(k) = fasr(k) + sum(fasr10tmp);
		end
	end
	[~,max_ef10] = max(ef10);
	[~,max_auc] = max(rocauc);
	[~,max_fasr10] = max(fasr);

	test_id = i;
	train_id = split;
	X_train = X(:,train_id)';
	X_test = X(:,test_id)';
	y_s_orig = allY(test_id,:);
	E = X_train;
	[m,n] = size(E);
	denominator = max(E) - min(E);
	intercept = min(E);
	E = (E - repmat(min(E),[m,1])) ./ repmat(denominator,[m,1]);
	mt = size(X_test,1);
	X_test = (X_test - repmat(intercept,[mt,1])) ./ repmat(denominator,[mt,1]);
	E = [E ones(m,1)];
	X_test = [X_test ones(mt,1)];
	n = n + 1;
	k = max_ef10;
	y_s = y_s_orig;
	[idx, C, sumd, D] = kmeans(E, k, 'Replicates', repeat);
	C = C(:,1:end-1);
	C = (C .* repmat(denominator,[k,1])) + repmat(intercept,[k,1]);
	tau = 1e-4;
	nOutcome = k;
	nGroupSelected = 0;
	while (nGroupSelected ~= features)
		tau = tau * 0.1;
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

	approach = probs' * C;
	G = G(1:end-1);
	approach(:,G) = x_test(:,1:end-1);
	informer_ef10{counter} = G;
	prediction_ef10{counter} = approach;
	clustersize_ef10(counter) = max_ef10;
	informer_rocauc{counter} = G;
	prediction_rocauc{counter} = approach;
	clustersize_rocauc(counter) = max_auc;
	informer_fasr10{counter} = G;
	prediction_fasr10{counter} = approach;
	clustersize_fasr10(counter) = max_fasr10;
	if (max_ef10 ~= max_auc)
		k = max_auc;
		y_s = y_s_orig;
		[idx, C, sumd, D] = kmeans(E, k, 'Replicates', repeat);
		C = C(:,1:end-1);
		C = (C .* repmat(denominator,[k,1])) + repmat(intercept,[k,1]);
		tau = 1e-4;
		nOutcome = k;
		nGroupSelected = 0;
		while (nGroupSelected ~= features)
			tau = tau * 0.1;
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

		approach = probs' * C;
		G = G(1:end-1);
		approach(:,G) = x_test(:,1:end-1);
		informer_rocauc{counter} = G;
		prediction_rocauc{counter} = approach;
		if (max_fasr10 == max_auc)
			informer_fasr10{counter} = G;
			prediction_fasr10{counter} = approach;
		end
	end
	if (max_fasr10 ~= max_auc && max_fasr10 ~= max_ef10)
		k = max_fasr10;
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

		approach = probs' * C;
		G = G(1:end-1);
		approach(:,G) = x_test(:,1:end-1);
		informer_fasr10{counter} = G;
		prediction_fasr10{counter} = approach;
	end

	counter = counter + 1;
end
