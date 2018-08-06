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
X=[atrue, Btrue];%]; %input vector 
X=X' ;
nrInput=size(X,1);
nrNodesHidden=[3];
nrOutput=1;
inputrange=[min(X); max(X)]';
X';
Networktype='ff';



NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,1000,'random');
% NetFF.trainalg='trainbp'
% NetFF.trainParam.mu=1e-7; 
%  NetFF.trainParam.mu_inc=0;
%  NetFF.trainParam.mu_dec=0;
NetFF.IW{1}=NetFF.IW{1}*1e-1;
% NetFF.LW=NetFF.LW*-1;
% NetFF.b{1}=-1*ones(size(NetFF.b{1}));
[NetFF,~]=trainNetwork(NetFF,Cm,X,1,[{'bo','wo','bi','wi'}]);



TRIeval = delaunayn(X');

figure
trisurf(TRIeval,X(1,:)',X(2,:)',Cm,'edgecolor','none');
hold on
plot3(X(1,:),X(2,:),out.yk,'.')