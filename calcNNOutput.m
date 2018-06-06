function Y2=calcNNOutput(NNset,X)
for h=1:size(NNset.centers,2) %loop over number of hidden layers
Nin=size(X,1);
L_end=size(X,2);
Nhidden=size(NNset.centers{h},1);
V1 = zeros(Nhidden,L_end);

for i=1:Nin
xij=X(i,:).*ones(size(V1));
% disp(NNset.centers(:,i));
cij=NNset.centers{h}(:,i)*ones(1,L_end);
wj=NNset.IW{h}(:,i);
V1=V1+(wj.*(xij-cij)).^2;
end;
%output for hidden layer
Y1=exp(-V1);
X=Y1; %output of current hidden layer is input for next hidden layer
end


%output of output layer
Y2=NNset.LW*Y1;

% hold on
% plot3(atrue,Btrue,Y2'-Cm,'.b');
end
