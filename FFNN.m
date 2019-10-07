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

%gradient descent (error back-propagation)
NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NNset.trainalg='traingd'; %gradient descent = error back propagation 
NNset.trainParam.mu=1e-5; 
NNset.trainParam.mu_inc=2;
NNset.trainParam.mu_dec=0.8;
NNset.trainParam.min_grad=1e-10;
trainp=NNset.trainParam;
trainal=NNset.trainalg;
if 0
do_sensitivity_ana;
end
% Levenberg - Marquardt

if 0
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

end


%% Find optimal number of neurons 

counter=0;
window=30;
El=[];
El_mean=[];
minE=inf;
nrit=5;
search=1;
n=5;
figure()
while search
    Emean=0;
    for k=1:nrit
        
        
        NN_c=createNNStructure(nrInput,n,nrOutput,inputrange,Networktype,1000,'random');
        NN_c.LW=randn(size(NN_c.LW))*0.01;
        NN_c.IW{1,1}=randn(size(NN_c.IW{1,1}));
        NN_c.b{1,1}=randn(size(NN_c.b{1,1}))*0.01;
        NN_c.b{2,1}=randn(size(NN_c.b{2,1}))*0.05;
        NN_c.trainalg=trainal; %gradient descent = error back propagation 
        NN_c.trainParam=trainp;
        NN_c.trainParam.epochs=1000;
        NN_c.trainParam.min_grad=1e-15;
        [~,E_i]=trainNetwork(NN_c,Y_train,X_train,X_val,Y_val,0,{'bo','wo','bi','wi'},0);
        Emean=Emean+(1/nrit)*E_i;
        El=[El;n,E_i];
    end
    El_mean=[El_mean;n,Emean];
    n=n+5;
    if Emean<minE
        minE=Emean;
        counter=0;
    else
        counter=counter+1;
    end
    cla();
    plot(El(:,1),El(:,2),'.')
    hold on
    plot(El_mean(:,1),El_mean(:,2))
    refreshdata()
    pause(0.01)
    if counter>=window
        search=1;
    end
end
save('fffindoptimum.mat','El','El_mean')          

