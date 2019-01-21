function [ret] = fasr10(z,y,scaffid)
	[len, p] = size(y);
	tenpercent = round(len / 10);
	shift = len - tenpercent;
	ret = [];
	for i=1:p
		all = length(unique(scaffid(find(y(:,i) == 1))));
		if all == 0
			ret = 1;
			continue;
		end
		tmp = [z(:,i),1 - y(:,i), y(:,i), scaffid];
		tmp = sortrows(tmp,2);
		tmp = sortrows(tmp,1);
		iden = find(tmp(shift+1:end,3) == 1) + shift;
		pred = length(unique(tmp(iden,4)));
		ret(i) = pred / all;
	end
end
