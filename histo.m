function [bin] = histo(data,bin_num)

edges = linspace(-1,1,bin_num);
min = data(1);
max = data(end);
data = data./abs(min);

[Y,E] = discretize(data,edges); % Y - bin for number, E - range 
bin = zeros(1,length(E));

for i = 1:length(data)
    if(isnan(Y(i)))
        Y(i) = 0+1;
        bin(Y(i)) = bin(Y(i))+1;
    else
       bin(Y(i)) = bin(Y(i))+1; 
    end
end
%bin(bin_num/2) = 0;
%bin(bin_num/2+1) = 0;
disp('Created histogram')
bar(edges,bin)
xlim([-1,1]);
%histogram(bin,edges)

end