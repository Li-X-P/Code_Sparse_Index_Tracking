function [x_final] = NNOMP_PDG(A,r, sparsity)
%%  
%   A: Daily returns of N assets
%   r: Daily returns of the market index
%   sparsity: Number of selected assets
%%
x_temp = algo_omp(A,r, sparsity);
x_final = x_temp;
index = find(x_temp~=0);
x_n0 = x_temp(index);
A_hat = A(:,index);
[x_out] = algo_allocation(A_hat, r, x_n0, 500);
x_out = x_out./sum(x_out);
x_final(index) = x_out;



function x = algo_omp(A,y, sparsity)
N = size(A,2);
x = zeros(N,1);
y = y(:);
r = y;
Ti = [];
iter = 1;
% norm of each column
norm_A = zeros(N,1);
for b = 1:N
    norm_A(b,1) = norm(A(:,b),2);
end
while(iter <= sparsity)
    ST = (A'*r) ./ norm_A;
    [~, b] = max(ST);
    if (max(ST) < 0)
        s1 = sprintf('The practical sparsity is less than the setting sparsity.');
        %         disp(s1);
        break;
    end
    Ti = [Ti b];
    At = A(:,Ti);
    %     s = (At' * At) \ At' * y;
    s = pinv(At,1e-8)* y;
    r = y - At*s;
    iter = iter + 1;
end
x(Ti) = s;

function x_out = algo_allocation(A, r,x_n0, maxiter)
x_out = x_n0;
iter2 = 0;
lambda = 0.5; 
stepsize = 5e-2; 
while(iter2 < maxiter)
    g = x_out;
    g(g(:)>0) = 1;
    g(g(:)<0) = -1;
    gra = 2*A'*(A*x_out -r)  +  2*lambda*(norm(x_out,1) - 1)*g;
    x_out = x_out - stepsize*gra;
    x_out(x_out<0) = 0;
    iter2 = iter2 + 1;
end
