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

% Cm=-1*Cm;
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


NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,10000,'random');
NetFF.trainalg='traingd';
NetFF.trainParam.mu=1; 
 NetFF.trainParam.mu_inc=10;
 NetFF.trainParam.mu_dec=0.1;

NetFF.LW=NetFF.LW/nrNodesHidden(1);
NetFF.b{1}=min(Cm)*ones(size(NetFF.b{1}));

[NetFF,~]=trainNetwork(NetFF,Y_train,X_train,X_val,Y_val,1,[{'bo','wo','wi','bi'}],0);



TRIeval = delaunayn(X_train');

figure
trisurf(TRIeval,X_train(1,:)',X_train(2,:)',Y_train,'edgecolor','none');
hold on
plot3(X(1,:),X(2,:),out.yk,'.')