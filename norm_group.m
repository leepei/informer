function norm_val = group_norm(x, group)
% Calculated group-wise norm for a matrix 'x'.
% The row indices belong to each group is stored in 'group' argument.

nGroup = length(group);
norm_val = zeros(nGroup,1);

for i=1:nGroup
  idx = group{i};
  norm_val(i) = norm(x(idx,:), 'fro');
end
