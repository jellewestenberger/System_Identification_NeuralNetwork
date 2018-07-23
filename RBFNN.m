%% Load Data
clear all
close all
load('atrue.mat');
load('Btrue.mat');
load('Vtrue.mat');
load('T.mat');
load('F16traindata_CMabV_2018','Cm');
X=[atrue, Btrue, Vtrue]; %input vector 


%% Create Initial Neural Network Structure
Networktype='rbf';      %choose network type: radial basis function (rbf) or feedforward (ff)
nrInput=size(X,2);      %number of inputs being used

nrOutput=1;             %Number of outputs
nrNodesHidden=[100];    %add columns to add more hidden layers;
nrHiddenlayers=size(nrNodesHidden,2);%number of nodes in the hidden layers
inputrange=[min(X); max(X)]'; 

NNset=createNNStructure(nrInput,nrHiddenlayers,nrNodesHidden,nrOutput,inputrange,Networktype);


%%---CHECK---- %% 
 check=NNCheck(NNset,nrInput,nrNodesHidden,nrOutput);


X=X';

%% Linear regression 
swit=1;
if swit
k=1; %only valid for one hidden layer (for now);
A=zeros(size(X,2),size(NNset.IW{k},1));
for j=1:size(A,2)
   vk=0;
    for i=1:size(X,1)
    vk=vk+(X(i,:)-NNset.centers{k}(j,i)).^2*(NNset.IW{k}(j,i))^2;
    end
   A(:,j)=exp(-vk').*NNset.LW(j);
end
a_est=inv(A'*A)*A'*Cm; %least-squared estimators 
Cm_est=A*a_est;
NNset_lin.a{k}=a_est; 
end


%% Levenberg Marquard
NNset=LevMar(NNset,Cm,X,10,0.1,1000,0);




