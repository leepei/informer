function [B] = SpaRSA_group(func_h, tau, B0, group, notS)
% SpaRSA for GroupLASSO with exclusion.
%
% func_h : function handle for objective and gradient
% tau    : penalty parameter for regularizer
% B0     : Initial point (matrix in a form of [B_1 B_2 ... B_K])
% group  : cell that containes the index of each group
% notS   : Initial set of groups that are not penalized.
%

  if tau <= 0
    error('tau should be strictly positive.');
  end

  [n,m] = size(B0);

  % SpaRSA parameters
  ETA = 2;
  ALPHA_MIN = 1e-4;
  ALPHA_MAX = 1e30;
  sigma = .1;

  ITER_MAX = 1e3;
  TOL      = 1e-8;
  iter = 0;

  nPartition = length(group);
  S          = setdiff([1:nPartition], notS);
  l1l2_mat = zeros(length(S), n);

  for i=1:length(S)
    l1l2_mat(i,group{S(i)}) = 1;
  end

  B = B0;
  obj = func_h(B) + tau*l1l2(B,l1l2_mat);
  while( iter<ITER_MAX )
    % choose alpha by Barzilai-Borwein
    [~,G] = func_h(B);
    if iter == 0
      alpha = ALPHA_MIN;
    else
      dG = G - G_prev;
      dB = B - B_prev;
      alpha = sum(sum(dB.*dG))/sum(sum(dB.*dB));
      alpha = max(ALPHA_MIN, min(ALPHA_MAX, alpha));
    end

    B_prev = B;
    G_prev = G;
    obj_prev = obj;
    while( alpha < ALPHA_MAX )
      U = B_prev - G_prev/alpha;
      beta = tau/alpha;

      for i=S(:)'
        subIdx = group{i};
        Usub = U(subIdx,:);
        THU  = max(norm(Usub,'fro') - beta, 0);
        B(subIdx,:) = Usub * (THU/(THU+beta));
      end

      for i=notS(:)'
        subIdx = group{i};
        B(subIdx,:) = U(subIdx,:);
      end

      obj = func_h(B) + tau*l1l2(B, l1l2_mat);
      val = obj_prev - sigma/2*alpha*norm(B-B_prev,'fro')^2;
      if obj < val
        break;
      end
      % fprintf('\r * [%05d] obj= %.3e, |df|= %.3e, alpha= %.3e', ...
      %         iter, obj, norm(G(:), inf), alpha);
      alpha = ETA*alpha;
    end
    if alpha > ALPHA_MAX
      obj = obj_prev;
      B   = B_prev;
      break;
    elseif abs((obj-obj_prev)/obj_prev) < TOL
      break;
    % elseif norm(B-B_prev,'fro')/norm(B,'fro') < TOL
    %   break;
    end
    % fprintf('\r * [%05d] obj= %.3e, |df|= %.3e, alpha= %.3e', ...
    %         iter, obj, norm(G(:), inf), alpha);

%    if mod(iter, 100) == 0
%      fprintf(' * [%05d] obj= %.3e, |df|= %.3e, alpha= %.3e\n', ...
%              iter, obj, norm(G(:), inf), alpha);
%       fprintf('\n');
%    end
    iter = iter+1;
  end

%  fprintf(' * [%05d] obj= %.3e, |df|= %.3e, alpha= %.3e\n', ...
%          iter, obj, norm(G(:), inf), alpha);
%  fprintf(' [Terminates in %5d iterations] obj= %.5e, |df|_inf= %.3e\n', ...
%          iter, obj, norm(G(:), inf));
   fprintf('\n [Terminates in %5d iterations] obj= %.5e, |df|_inf= %.3e\n', ...
           iter, obj, norm(G(:), inf));
end

function [w] = l1l2(B, l1l2_mat)
  [n,m] = size(B);
  w = sum( sqrt( l1l2_mat*sum(B.^2, 2) ));
end
