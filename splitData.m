function [X_train,X_val,Y_train,Y_val] = splitData(X,Y,fr_train,fr_val,ran)
N=size(X,1);

if fr_train==1
    X_train=X;
    X_val=X;
    Y_train=Y;
    Y_val=Y;

else

    if ran
       s = RandStream('mt19937ar','Seed',1); %fix seed for consistent results in the report
       i=randperm(s,N);
       X=X(i,:);
       Y=Y(i);   
    end
    % i_total=1:N;
    % i_train=1:floor(1/fr_train):N;
    % i_val=setdiff(i_total,i_train);
    % 
    i_train=floor(fr_train*N);
    i_val=floor(fr_val*N)+i_train;
    i_val=i_train+1:i_val+1;
    i_train=1:i_train;

    X_train=X(i_train,:);
    X_val=X(i_val,:);
    Y_train=Y(i_train);
    Y_val=Y(i_val);
end

end