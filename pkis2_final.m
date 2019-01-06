k = 6;
load PKIS2_1um;

l = length(targets);
features = 16;
repeat = 100;
n = size(X,1)
nFeature = n;
group = cell(nFeature+1,1);
for i=1:nFeature
	group{i} = [i]; %each group has 1 coordinate, by nature the sparsa code groups the coordinates in one group for all classes together
end
group{nFeature + 1} = n + 1;
G0 = [nFeature + 1];%do not penalize the bias term
nGroup = length(group);
nPreselectedGroup = length(G0);

X_train = X';
E = X_train;
[m,n] = size(E);
denominator = max(E) - min(E);
intercept = min(E);
E = (E - repmat(min(E),[m,1])) ./ repmat(denominator,[m,1]);
E = [E ones(m,1)];
n = n + 1;
[idx, C, sumd, D] = kmeans(E, k, 'Replicates', repeat);
%{C = C(:,1:end-1);
C = (C .* repmat(denominator,[k,1])) + repmat(intercept,[k,1]);
tau = 1e-5;
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
idx_feature = sort([group{G}])
x_train = E(:,idx_feature);
fn = @(B)func_mlr(B, x_train', idx, 0);
Bsub = B(idx_feature,:);
[Bsub] = LBFGS_MLE(fn, Bsub);%}
