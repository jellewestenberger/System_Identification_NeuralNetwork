function [A,theta,exps]=OLSQ_est(order,X,Y,type)
nrvars=size(X,2);

if strcmp(type,'simple')
A=zeros(size(X,1),order+1);
A(:,1)=1;

for i=1:order 
    A(:,i+1)=(sum(X,2)).^i;
end
elseif strcmp(type,'allorder')

    ordl=0:order;
    ordl=(ordl.*ones(size(ordl,2),nrvars)')';

    exps=ordl(1,:);
    %find all combinations of exponentials 
    while(sum(ordl(1,:)==order)<nrvars) 
       a=circshift(ordl,-1);
       if ordl(1,1)==order
           nr=sum(ordl(1,1:(end-1))==order);
           ordl(:,1:nr+1)=a(:,1:nr+1);
       else
           ordl(:,1)=a(:,1);
       end
       exps=[exps;ordl(1,:)];   
    end
    A=x2fx(X,exps);


elseif strcmp(type,'sumorder')
    exps=zeros(1,nrvars);
    for k=1:order
    exps=[exps;exponentials(nrvars,k)];
    end
    A=x2fx(X,exps);
end
theta=(pinv(A'*A))*A'*Y;

function exps=exponentials(vars,order)
if vars<=1
    exps=order;
    
else
    exps=zeros(0,size(vars,2));
    
    for j=order:-1:0
        rc=exponentials(vars-1,order-j);
        exps=[exps;j*ones(size(rc,1),1),rc];
    end
end
end

end