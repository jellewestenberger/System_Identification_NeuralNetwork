%% this script is used to check the functionality of the LM implementation
%   It is used to approximate a simple sine function.

clear all
close all
u=linspace(0,20,500)';
v=linspace(0,33,500)';
w=linspace(0,50,500)';
y=sin(u)+5*cos(3*v)+sin(w-0.5*pi);

X=[u, v, w];


n_input=size(X,2);
n_hidden=1;
n_nodes=2000;

n_output=1;
input_range=[min(X);max(X)]'
type='rbf'
X=X';

network=createNNStructure(n_input,n_hidden,n_nodes,n_output,input_range,type);
network.a{1}=ones(n_nodes,1);
mu_inc=10;
mu_dec=0.1;

network=LevMar(network,y,X,mu_inc,mu_dec,100,1,[0,0,1,0
    ]);



