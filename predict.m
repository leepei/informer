function [x] = predict(y,Bsub,C,G)
	[l,n] = size(y);
	k = size(Bsub,2);
	if (n == size(Bsub,1) - 1)
		y = [y ones(l,1)];
	end
	Bx = y * Bsub;
	Bx = Bx';
	[nBx, mBx] = size(Bx);
	[max_val, max_idx] = max(Bx);
	Bx_mod = Bx-repmat(max_val, nBx, 1);

	exp_Bx     = exp(Bx_mod);
	sum_exp_Bx = sum(exp_Bx);
	log_sum_exp_Bx = max_val + log(sum_exp_Bx);
	probs = exp(Bx - repmat(log_sum_exp_Bx, k, 1));

	x = probs' * C;
	G = G(1:end-1);
	x(:,G) = y(:,1:end-1);
end
