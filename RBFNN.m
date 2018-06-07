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

nrOutput=1; 
nrNodesHidden=[100]; %add columns to add more hidden layers;
nrHiddenlayers=size(nrNodesHidden,2);%number of nodes in the hidden layers
inputrange=[min(X); max(X)]';


NNset=createNNStructure(nrInput,nrHiddenlayers,nrNodesHidden,nrOutput,inputrange,Networktype);
  
%% ---CHECK---- %% 
 check=NNCheck(NNset,nrInput,nrNodesHidden,nrOutput);
%% ------------ %% 

X=X';

% Nin=size(X

%% Linear regression 
%not that this is currently only valid for a 1 hidden layer network
Phis=zeros(size(X,2),nrNodesHidden);

A=zeros(size(X,2),nrNodesHidden);
a=zeros(nrNodesHidden,1);

for j=1:nrNodesHidden %node identifier 

eval=0;
for i=1:size(X,1) %input identifier (alpha, beta or V) 
x=X(i,:);
w=NNset.IW{1,1}(j,i); %input weight;
c=NNset.centers{1,1}(j,i); %centers
eval=eval+(w^2)*(x-c).^2;     %input variable for activation function (consists of sum of input weights multiplied with corresponding center differences)       
end
Phis(:,j)=exp(-eval)';
end
A(:,:)=Phis.*NNset.LW;
a=((A'*A)^(-1))*A'*Cm;
Cm_estimated=A*a;
NNset.a=a;
%%
%% Creating output---

[Y, hiddenoutput]=calcNNOutput(NNset,X); %nice for double checking with Cm_estimated (should be exactly equal)

%% Levenberg Marquard
%calculate dE/d Wjk   (Wjk = output weight);
ek=Y'-Cm;
h=1
dEdWjk{h}=zeros(size(NNset.IW{h},1),1);

for j=1:size(Wjk{h},1)
yj=hiddenoutput{1}(j,:);    
dEdWjk{h}(j)=sum(ek.*yj');
end


%% plotting 
figure
plot3(atrue,Btrue,Y','.b'); 
hold on
plot3(atrue,Btrue,Cm_estimated,'.');
hold on
plot3(atrue,Btrue,Cm,'.k');
%%