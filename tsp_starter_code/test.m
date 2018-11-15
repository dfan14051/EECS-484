A=500.0;
B=500.0;
C=200.0;
D=500.0;
cost = zeros([100,1]); 
parfor i=1:numel(cost)
    [sumV,tripcost]=hopfield_TSP(A,B,C,D);
    cost(i) = tripcost;
end
histogram(cost)
minimum_ = min(cost)
maximum_ = max(cost)
mean_ = mean(cost)
median_ = median(cost)
std_ = std(cost)