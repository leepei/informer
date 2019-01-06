function [B] = LBFGS_MLE(func_h, B0)
% LBFGS for Multiclass Logistic Regression
%
% func_h : function handle for objective and gradient
% B0 : Initial solution
%

  ITER_MAX = 5e5;
  STEP_MIN = 1e-7;
  TOLER    = 1e-6;
  PAIRS    = 10; % the number of recent pairs (s,y) to keep

  [nB, mB] = size(B0);

  % norm for df
  norm_df = @(x)norm(x(:),inf);

  x_   = B0;

  norm_x_ = norm(x_, 'fro');
  [obj_, df_] = func_h(x_);
  obj_ = obj_;
  df_  = df_;

  s_arr = zeros(nB, mB, PAIRS);
  y_arr = zeros(nB, mB, PAIRS);

  it = 0;
%  fprintf(1, ' Initialized....  f=%.5e\n', obj_ );
  step = 1;
  while( it < ITER_MAX && norm_df(df_) > TOLER && step > STEP_MIN )
    [x, obj, df] = deal(x_, obj_, df_);
    p  = -LBFGS(s_arr, y_arr, df, it, min(it, PAIRS));

    step = 1;
    while(step > STEP_MIN)
      x_   = x + step*p;
      norm_x_ = norm(x_, 'fro');
      if ( func_h(x_) <= obj + 0.01*step*InnerProd(df, p)); break; end;
      step = step/2;
    end
    norm_x_ = norm(x_, 'fro');
    [obj_, df_] = func_h(x_);
    obj_ = obj_;
    df_  = df_;

    % Update the set of pairs (s,y)
    col_idx = mod(it,PAIRS)+1;
    s_arr(:,:,col_idx) = x_-x;
    y_arr(:,:,col_idx) = df_-df;

    it = it + 1;
    if mod(it,1000) == 0 || norm_df(df_) <= TOLER || step <= STEP_MIN
      fprintf(1, ' * [%05d]: obj= %.5e, |df|= %.3e, step= %.2e\n', ...
                 it, obj_, norm_df(df_), step );
	end
%      fprintf(1, '\r * [%05d]: obj= %.5e, |df|= %.3e, step= %.2e\n', ...
%                  it, obj_, norm_df(df_), step );
%     else
%       fprintf(1, '\r * [%05d]: obj= %.5e, |df|= %.3e, step= %.2e', ...
%                  it, obj_, norm_df(df_), step );
%    end
  end
  fprintf(1, ' [Terminates in %5d iterations] obj= %.5e, |df|_inf= %.3e\n', ...
             it, obj_, norm_df(df_) );

  B = x_;
end

function [ip] = InnerProd(x,y)
  ip = sum(sum(x.*y));
end

function [p] = LBFGS(s, y, df, it, m)
%
% Inputs
%   s = [ s_{k-1} s_{k-2} ... s_{k-m} ]         s_i =  x_{i+1} -  x_{i}
%   y = [ y_{k-1} y_{k-2} ... y_{k-m} ]         y_i = df_{i+1} - df_{i}
%  df = df_k
%
% Algorithm 7.4 from Nocedal & Wright (2006)
%
%            s_{k-1}'*y_{k-1}
%       H0 = ---------------- I
%            y_{k-1}'*y_{k-1}
%
%
% Created: 24/Jun/2014
% tdkim@cs.wisc.edu
%

  if m < 1
    p = df / sqrt(InnerProd(df,df));
    return;
  end

  q = df;
  alfa = zeros(m,1); rho  = zeros(m,1);

  lst_idx = circshift([m:-1:1], [0 mod(it,m)]);
  for i=lst_idx
    rho(i)  = 1/InnerProd(y(:,:,i), s(:,:,i));
    alfa(i) = rho(i)*InnerProd(s(:,:,i), q);
    q       = q - alfa(i)*y(:,:,i);
  end

  idx1 = lst_idx(1);
  p = InnerProd(s(:,:,idx1), y(:,:,idx1))/InnerProd(y(:,:,idx1), y(:,:,idx1))*q;

  lst_idx = circshift([1:m], [0 -mod(it,m)]);
  for i=lst_idx
    beta = rho(i)*InnerProd(y(:,:,i), p);
    p    = p + s(:,:,i)*(alfa(i) - beta);
  end
end
