%% Load Data
clear all
close all
load('atrue.mat');
load('Btrue.mat');
load('Vtrue.mat');
load('T.mat');
load('F16traindata_CMabV_2018','Cm');
% load_f16data2018

load('NNset.mat');

atrue_nom=normalize(atrue,'zscore');
btrue_nom=normalize(Btrue,'zscore');
fr_train=0.7;
fr_val=1-fr_train;
[X_train,X_val,Y_train,Y_val]=splitData([atrue_nom,btrue_nom],Cm,fr_train,fr_val,1);


% X=[atrue,Btrue];
% X=[atruenom, btruenom];%]; %input vector 
% X=X' ;
nrInput=size(X_train,2);
nrNodesHidden=[100];
nrOutput=1;
inputrange=[min(X_train); max(X_train)]';
% X';
Networktype='ff';
X_train=X_train';
X_val=X_val';

%% Sensitivity Initial Parameters 
NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NNset.trainalg='traingd'; %gradient descent = error back propagation 
NNset.trainParam.mu=1e-5; 
NNset.trainParam.mu_inc=2;
NNset.trainParam.mu_dec=0.8;
NNset.trainParam.min_grad=1e-10;

if 1
do_sensitivity_ana;
end
%%
NNset=createNNStructure(nrInput,100,nrOutput,inputrange,Networktype,4000,'random');
NNset.LW=randn(size(NNset.LW))*0.01;
NNset.IW{1,1}=randn(size(NNset.IW{1,1}));
NNset.b{1,1}=randn(size(NNset.b{1,1}))*0.01;
NNset.b{2,1}=randn(size(NNset.b{2,1}))*0.002;
NNset.trainalg='trainlm';
NNset.trainParam.mu=1e5; 
NNset.trainParam.mu_inc=10;
NNset.trainParam.mu_dec=0.05;
NNset.trainParam.min_grad=1e-10;

do_sensitivity_ana;
