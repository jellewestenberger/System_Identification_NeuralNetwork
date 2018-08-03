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
nrNodesHidden=[100,100];
nrOutput=1;
inputrange=[min(X); max(X)]';
X';
Networktype='ff';



NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,'ones');
out=calcNNOutput(NetFF,X);

TRIeval = delaunayn(X');

figure
trisurf(TRIeval,X(1,:)',X(2,:)',Cm,'edgecolor','none');
hold on
plot3(X(1,:),X(2,:),out.yk,'.')