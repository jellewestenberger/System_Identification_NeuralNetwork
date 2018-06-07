function [Y2, hiddenoutput,Vj]=calcNNOutput(NNset,X)
hiddenoutput={};
Vj={};
for k=1:size(NNset.centers,2) %loop over number of hidden layers
    hiddenoutput{k}=zeros(size(NNset.IW{k},1),size(X,2));
    
    Nin=size(X,1);
    L_end=size(X,2);
    Nhidden=size(NNset.centers{k},1);
    V1 = zeros(Nhidden,L_end);

    for i=1:Nin
        xij=X(i,:).*ones(size(V1));
        % disp(NNset.centers(:,i));
        cij=NNset.centers{k}(:,i)*ones(1,L_end);
        wj=NNset.IW{k}(:,i);
        V1=V1+(wj.*(xij-cij)).^2;
    end;
    %output for hidden layer
    Y1=NNset.a{k}(:,k).*exp(-V1);
    Vj{k}=V1;
    hiddenoutput{k}(:,:)=Y1;
    X=Y1; %output of current hidden layer is input for next hidden layer
end


%output of output layer
Y2=NNset.LW*Y1;

% hold on
% plot3(atrue,Btrue,Y2'-Cm,'.b');
end
