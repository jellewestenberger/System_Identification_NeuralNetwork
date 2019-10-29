%% this script is used to verify the functionality of the Neuralnetwork training
%   It attempts to approximate the sinusoidal function below

clear all
close all
u=linspace(0,20,500)';
v=linspace(0,33,500)';
w=linspace(0,50,500)';
y=sin(u)+5*cos(3*v)+sin(w-0.5*pi);

X=[u, v, w];


n_input=size(X,2);
n_hidden=1;
n_nodes=150;

n_output=1;
input_range=[min(X);max(X)]';
type='rbf';
X=X';

% Nset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,epoch,inittype)

network=createNNStructure(n_input,n_nodes,n_output,input_range,type,inf,'random');
network.a{1}=ones(n_nodes,1);
network.trainalg='trainlm';
network.trainParam.mu=0.1;
mu_inc=10;
mu_dec=0.1;

% trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plotf,selector,optimizeorder)

[NNsetmin, minerror,El,evl]=trainNetwork(network,y,X,X,y,1,{'a','c','wo','wi'},0);
calc


