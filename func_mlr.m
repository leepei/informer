function [f, df] = func_mlr(B, x, y, lambda)
% Returns Maximum Likelihood Estimation objective and its gradient.
% x : Observation vecters
% y : Outcomes corresponding to each observation

  if (nargin < 4)
	  lambda = 2^(-10);
  end
  Bx = B'*x;
  [nBx, mBx] = size(Bx);
  [max_val, max_idx] = max(Bx);
  Bx_mod = Bx-repmat(max_val, nBx, 1);

  exp_Bx     = exp(Bx_mod);
  sum_exp_Bx = sum(exp_Bx);
  log_sum_exp_Bx = max_val + log(sum_exp_Bx);

  vec_f = log_sum_exp_Bx - sum(B(:,y).*x);
  f = sum( vec_f ) + norm(B,'fro')^2 * lambda / 2;

  if nargout>1
    y_mat = sparse([1:mBx], y, ones(mBx,1));
    term1 = x*y_mat;

    term2 = x*(exp_Bx*spdiags(1./sum_exp_Bx(:), 0, mBx, mBx))';
    df = term2 - term1 + B * lambda;
  end
end

