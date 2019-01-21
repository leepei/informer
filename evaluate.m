function [ef10,auc] = evaluate(z,y)
	[len, p] = size(y);
	tenpercent = round(len / 10);
	base = sum(y);
	pairs = base .* (len - base);
	ef10max = min(base, tenpercent);
	ef10max = ef10max ./ base * (len / tenpercent);
	correct = zeros(p,1);
	for i=1:p
		if base(i) == 0
			ef10max(i) = 2;
			ef10(i) = 2;
			pairs(i) = 1;
			correct(i) = pairs(i);
			continue;
		end

		tmp = [z(:,i),1 - y(:,i), y(:,i)];
		tmp = sortrows(tmp,2);
		tmp = sortrows(tmp,1);
		ef10(i) = sum(tmp(end-tenpercent+1:end, 3)) / base(i) * (len/tenpercent);

		counter1 = base(i);
		counter2 = len - base(i);
		for j=len:-1:1
			current = tmp(j,3);
			counter1 = counter1 - current;
			counter2 = counter2 - (1 - current);
			correct(i) = correct(i) + current * counter2;
			if (counter1 <= 0)
				break;
			end
		end
	end
	ef10 = 1 + (ef10 - 1) ./ (ef10max - 1);
	ef10 = ef10 / 2;
	auc = correct ./ pairs';
end
