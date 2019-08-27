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
%set 1
rng(50);
if 1
El=[];
Eprop={};
for k =1:5
NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NetFF.LW=randn(size(NetFF.LW))*0.01;
NetFF.IW{1,1}=randn(size(NetFF.IW{1,1}));
NetFF.b{1,1}=randn(size(NetFF.b{1,1}))*0.01;
NetFF.b{2,1}=randn(size(NetFF.b{2,1}))*0.01;
NetFF.trainalg='traingd'; %gradient descent = error back propagation 
NetFF.trainParam.mu=1e-5; 
NetFF.trainParam.mu_inc=2;
NetFF.trainParam.mu_dec=0.8;
NetFF.trainParam.min_grad=1e-10;
[NetFF,Ei,Elist,evl]=trainNetwork(NetFF,Y_train,X_train,X_val,Y_val,0,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)];
Eprop{k}=Elist;
end
save('FFset1','El','Eprop','evl');
end

if 1
%set 2
Eprop={};
El=[];
for k =1:5
NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NetFF.LW=randn(size(NetFF.LW))*0.01;
NetFF.IW{1,1}=5*randn(size(NetFF.IW{1,1}));
NetFF.b{1,1}=randn(size(NetFF.b{1,1}))*0.01;
NetFF.b{2,1}=randn(size(NetFF.b{2,1}))*0.01;
NetFF.trainalg='traingd'; %gradient descent = error back propagation 
NetFF.trainParam.mu=1e-5; 
NetFF.trainParam.mu_inc=2;
NetFF.trainParam.mu_dec=0.8;
NetFF.trainParam.min_grad=1e-10;
[NetFF,Ei,Elist,evl]=trainNetwork(NetFF,Y_train,X_train,X_val,Y_val,0,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)];
Eprop{k}=Elist;
end
save('FFset2','El','Eprop','evl');
end
%set 3
if 1
Eprop={};
El=[];
for k =1:5
NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NetFF.LW=randn(size(NetFF.LW))*0.01;
NetFF.IW{1,1}=0.2*randn(size(NetFF.IW{1,1}));
NetFF.b{1,1}=randn(size(NetFF.b{1,1}))*0.01;
NetFF.b{2,1}=randn(size(NetFF.b{2,1}))*0.01;
NetFF.trainalg='traingd'; %gradient descent = error back propagation 
NetFF.trainParam.mu=1e-5; 
NetFF.trainParam.mu_inc=2;
NetFF.trainParam.mu_dec=0.8;
NetFF.trainParam.min_grad=1e-10;
[NetFF,Ei,Elist,evl]=trainNetwork(NetFF,Y_train,X_train,X_val,Y_val,0,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)];
Eprop{k}=Elist;
end
save('FFset3','El','Eprop','evl');
end
%set 4
if 1
Eprop={};
El=[];
for k =1:5
NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NetFF.LW=randn(size(NetFF.LW))*0.05;
NetFF.IW{1,1}=randn(size(NetFF.IW{1,1}));
NetFF.b{1,1}=randn(size(NetFF.b{1,1}))*0.01;
NetFF.b{2,1}=randn(size(NetFF.b{2,1}))*0.01;
NetFF.trainalg='traingd'; %gradient descent = error back propagation 
NetFF.trainParam.mu=1e-5; 
NetFF.trainParam.mu_inc=2;
NetFF.trainParam.mu_dec=0.8;
NetFF.trainParam.min_grad=1e-10;
[NetFF,Ei,Elist,evl]=trainNetwork(NetFF,Y_train,X_train,X_val,Y_val,0,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)];
Eprop{k}=Elist;
end
save('FFset4','El','Eprop','evl');
end

%set 5
if 1
Eprop={};
El=[];
for k =1:5
NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NetFF.LW=randn(size(NetFF.LW))*0.002;
NetFF.IW{1,1}=randn(size(NetFF.IW{1,1}));
NetFF.b{1,1}=randn(size(NetFF.b{1,1}))*0.01;
NetFF.b{2,1}=randn(size(NetFF.b{2,1}))*0.01;
NetFF.trainalg='traingd'; %gradient descent = error back propagation 
NetFF.trainParam.mu=1e-5; 
NetFF.trainParam.mu_inc=2;
NetFF.trainParam.mu_dec=0.8;
NetFF.trainParam.min_grad=1e-10;
[NetFF,Ei,Elist,evl]=trainNetwork(NetFF,Y_train,X_train,X_val,Y_val,0,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)];
Eprop{k}=Elist;
end
save('FFset5','El','Eprop','evl');
end

%set 6
if 1
Eprop={};
El=[];
for k =1:5
NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NetFF.LW=randn(size(NetFF.LW))*0.01;
NetFF.IW{1,1}=randn(size(NetFF.IW{1,1}));
NetFF.b{1,1}=randn(size(NetFF.b{1,1}))*0.05;
NetFF.b{2,1}=randn(size(NetFF.b{2,1}))*0.01;
NetFF.trainalg='traingd'; %gradient descent = error back propagation 
NetFF.trainParam.mu=1e-5; 
NetFF.trainParam.mu_inc=2;
NetFF.trainParam.mu_dec=0.8;
NetFF.trainParam.min_grad=1e-10;
[NetFF,Ei,Elist,evl]=trainNetwork(NetFF,Y_train,X_train,X_val,Y_val,0 ,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)];
Eprop{k}=Elist;
end
save('FFset6','El','Eprop','evl');
end
%set 7
if 1
Eprop={};
El=[];
for k =1:5
NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NetFF.LW=randn(size(NetFF.LW))*0.01;
NetFF.IW{1,1}=randn(size(NetFF.IW{1,1}));
NetFF.b{1,1}=randn(size(NetFF.b{1,1}))*0.002;
NetFF.b{2,1}=randn(size(NetFF.b{2,1}))*0.01;
NetFF.trainalg='traingd'; %gradient descent = error back propagation 
NetFF.trainParam.mu=1e-5; 
NetFF.trainParam.mu_inc=2;
NetFF.trainParam.mu_dec=0.8;
NetFF.trainParam.min_grad=1e-10;
[NetFF,Ei,Elist,evl]=trainNetwork(NetFF,Y_train,X_train,X_val,Y_val,0,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)];
Eprop{k}=Elist;
end
save('FFset7','El','Eprop','evl');
end
%set 8
if 1
Eprop={};
El=[];
for k =1:5
NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NetFF.LW=randn(size(NetFF.LW))*0.01;
NetFF.IW{1,1}=randn(size(NetFF.IW{1,1}));
NetFF.b{1,1}=randn(size(NetFF.b{1,1}))*0.01;
NetFF.b{2,1}=randn(size(NetFF.b{2,1}))*0.05;
NetFF.trainalg='traingd'; %gradient descent = error back propagation 
NetFF.trainParam.mu=1e-5; 
NetFF.trainParam.mu_inc=2;
NetFF.trainParam.mu_dec=0.8;
NetFF.trainParam.min_grad=1e-10;
[NetFF,Ei,Elist,evl]=trainNetwork(NetFF,Y_train,X_train,X_val,Y_val,0,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)];
Eprop{k}=Elist;
end
save('FFset8','El','Eprop','evl');
end
%set 9
if 1
Eprop={};
El=[];
for k =1:5
NetFF=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NetFF.LW=randn(size(NetFF.LW))*0.01;
NetFF.IW{1,1}=randn(size(NetFF.IW{1,1}));
NetFF.b{1,1}=randn(size(NetFF.b{1,1}))*0.01;
NetFF.b{2,1}=randn(size(NetFF.b{2,1}))*0.002;
NetFF.trainalg='traingd'; %gradient descent = error back propagation 
NetFF.trainParam.mu=1e-5; 
NetFF.trainParam.mu_inc=2;
NetFF.trainParam.mu_dec=0.8;
NetFF.trainParam.min_grad=1e-10;
[NetFF,Ei,Elist,evl]=trainNetwork(NetFF,Y_train,X_train,X_val,Y_val,0,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)];
Eprop{k}=Elist;
end
save('FFset9','El','Eprop','evl');
end
%%