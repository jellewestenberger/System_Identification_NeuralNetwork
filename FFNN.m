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
atruenom=normalize(atrue,'zscore');
btruenom=normalize(Btrue,'zscore');



% X=[atrue,Btrue];
X=[atruenom, btruenom];%]; %input vector 
X=X' ;
nrInput=size(X,1);
nrNodesHidden=[150];
nrOutput=1;
inputrange=[min(X); max(X)]';
X';
Networktype='ff';



NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,10000,'random');
NetFF.trainalg='trainlm';
% NetFF.trainParam.mu=1e-7; 
%  NetFF.trainParam.mu_inc=0;
%  NetFF.trainParam.mu_dec=0;

NetFF.LW=NetFF.LW/nrNodesHidden(1);
NetFF.b{1}=min(Cm)*ones(size(NetFF.b{1}));
% NetFF.b{1}=-1*ones(size(NetFF.b{1}));
% load 'NNset.mat'
% NetFF=NNset;

[NetFF,~]=trainNetwork(NetFF,Cm,X,1,[{'bi','wi','wo'}]);



TRIeval = delaunayn(X');

figure
trisurf(TRIeval,X(1,:)',X(2,:)',Cm,'edgecolor','none');
hold on
plot3(X(1,:),X(2,:),out.yk,'.')