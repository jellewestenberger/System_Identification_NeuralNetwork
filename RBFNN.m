%% Load Data
clear all
close all
load('atrue.mat');
load('Btrue.mat');
load('Vtrue.mat');
load('T.mat');
load('F16traindata_CMabV_2018','Cm');
load_f16data2018
% Cm=-1*Cm;
X=[atrue, Btrue,Vtrue]; %input vector 
% X=[alpha_m,beta_m];

%% Create Initial Neural Network Structure
Networktype='rbf';      %choose network type: radial basis function (rbf) or feedforward (ff)
nrInput=size(X,2);      %number of inputs being used
nrOutput=1;             %Number of outputs
nrNodesHidden=[100];    %add columns to add more hidden layers;
inputrange=[min(X); max(X)]'; 

NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,'ones');


%%---CHECK---- %% 
 check=NNCheck(NNset,nrInput,nrNodesHidden,nrOutput);


X=X';

%% Linear regression 
NNset_lin=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,'random');
swit=1;
if swit
k=1; %only valid for one hidden layer (for now);
A=zeros(size(X,2),size(NNset_lin.IW{k},1));
for j=1:size(A,2)
   vk=0;
    for i=1:size(X,1)
    vk=vk+(X(i,:)-NNset_lin.centers{k}(j,i)).^2*(NNset_lin.IW{k}(j,i))^2;
    end
   A(:,j)=exp(-vk').*NNset_lin.LW(j);
end
a_est=inv(A'*A)*A'*Cm; %least-squared estimators 
Cm_est=A*a_est;
NNset_lin.a{k}=a_est; 
end
result=calcNNOutput(NNset_lin,X);

TRIeval = delaunayn(X(1:2,:)');

figure
trisurf(TRIeval,X(1,:)',X(2,:)',Cm,'edgecolor','none');
hold on
plot3(X(1,:),X(2,:),result.yk,'.')

%% Levenberg Marquard
 [NNset, ~]=LevMar(NNset,Cm,X,10,0.1,1000,1,[1,1,1,1]);
 result=calcNNOutput(NNset,X);

%% golden ratio search:
GR=(1+sqrt(5))/2.; 
a=1;
b=1000;
c=b-((b-a)/GR);
d=a+((b-a)/GR);
c=floor(c); %we need integers for number of neurons
d=floor(d);

while abs(c-d)>1
    
NN_c=createNNStructure(nrInput,[floor(c)],nrOutput,inputrange,Networktype,'ones');   
NN_d=createNNStructure(nrInput,[d],nrOutput,inputrange,Networktype,'ones');   
[~,E_c]=LevMar(NN_c,Cm,X,10,0.1,100,0,[1,0,0,0]);
[~,E_d]=LevMar(NN_d,Cm,X,10,0.1,100,0,[1,0,0,0]);
    if E_c< E_d
        b=d;
    else
        a=c;
    end
    c=round(b-((b-a)/GR),0);
    d=round(a+((b-a)/GR),0);

end



TRIeval = delaunayn(X');

figure
trisurf(TRIeval,X(1,:)',X(2,:)',Cm,'edgecolor','none');
hold on
plot3(X(1,:),X(2,:),result.yk,'.')


