function [s] = logtrick2(lA, dim)
% logtrick - "log sum trick" - calculate log(sum(A)) using only log(A) 
%
%   s = logtrick(lA, dim)
%
%   lA = matrix of log values
%
%   if lA is a matrix, then the log sum is calculated over dimension dim.
% 

[mv, mi] = max(lA, [], dim);
rr = size(lA);
rr(dim) = 1;
temp = bsxfun(@minus, lA, reshape(mv, rr));
cterm = sum(exp(temp),dim);
s = mv + log(cterm);
end

