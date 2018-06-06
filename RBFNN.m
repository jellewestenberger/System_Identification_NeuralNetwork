clear all
close all
load('atrue.mat');
load('Btrue.mat');
load('Vtrue.mat');
load('NetExampleRBF.mat');
load('T.mat');
load('F16traindata_CMabV_2018','Cm');
X=[atrue, Btrue, Vtrue]; %input vector 

Networktype='rbf'; %choose network type: radial basis function (rbf) or feedforward (ff)
nrInput=size(X,2);     %number of inputs being used
nrHiddenlayers=2; 
nrOutput=1; 
nrNodesHidden=[6 4];%number of nodes in the hidden layers
inputrange=[min(X); max(X)]';

NNset=createNNStructure(nrInput,nrHiddenlayers,nrNodesHidden,nrOutput,inputrange,Networktype);

%---CHECK----
 check=NNCheck(NNset,nrInput,nrNodesHidden,nrOutput);
%------------



% Nin=size(X
%Creating output---
X=X';
Y=calcNNOutput(NNset,X);


figure
plot3(atrue,Btrue,Y','.');
hold on
plot3(atrue,Btrue,Cm,'.k');