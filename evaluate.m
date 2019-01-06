function [auc] = evaluate(z,y)
	[len, p] = size(y);
	auc = 0;
	for i=1:p
		tmp = [z(:,i),1 - y(:,i), y(:,i)];
		if length(unique(z(:,i))) == 1
			auc = auc + 0;
			continue;
		end
		tmp = sortrows(tmp,2);
		tmp = sortrows(tmp,1);
		weight = [1:len];
		best = sum(y(:,i) == 1);
		if best == 0
			auc = auc + 1;
			continue;
		end
		best_performance = weight * sort(y(:,i));
		auc = auc + weight * tmp(:,3) / sum(best_performance);
	end
end

