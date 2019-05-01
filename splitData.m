function [X_train,X_val,Y_train,Y_val] = splitData(X,Y,fr_train,fr_val,ran)
N=size(X,1);
if ran
   i=randperm(N);
   X=X(i,:);
   Y=Y(i);   
end

i_train=floor(fr_train*N);
i_val=floor(fr_val*N)+i_train;
i_val=i_train+1:i_val+1;
i_train=1:i_train;

X_train=X(i_train,:);
X_val=X(i_val,:);
Y_train=Y(i_train);
Y_val=Y(i_val);

end