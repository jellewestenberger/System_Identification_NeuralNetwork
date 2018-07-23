%% this script is used to check the functionality of the LM implementation
%   It is used to approximate a simple sine function.

clear all
close all
u=linspace(0,20,500)';
v=linspace(0,33,500)';
w=linspace(0,50,500)';
y=sin(u)+20*cos(3*v)+sin(w-0.5*pi);
y=y'
X=[u, v, w];


n_input=3;
n_hidden=1;
n_nodes=5000;
n_output=1;
input_range=[min(X);max(X)]'
type='rbf'
X=X';

network=createNNStructure(n_input,n_hidden,n_nodes,n_output,input_range,type);
network.a{1}=ones(n_nodes,1);
mu1=1;
mu2=1;
mu_inc=10;
mu_dec=0.1;
E2old=inf;

for eval=1:500
close all
nnoutput=calcNNOutput(network,X);



ekq=y-nnoutput.yk;
E1=sum(0.5*ekq.^2);
disp(E1);
plot(y);
hold on
plot(nnoutput.yk);
legend('original','approximated');
pause

dEdWjk=nnoutput.yi{1,2}*ekq'*(-1);



J=dEdWjk;
d=LM(J,E1,mu1);
network.LW=network.LW-d;

nnoutput=calcNNOutput(network,X);
ekq=y-nnoutput.yk;
E2=sum(0.5*ekq.^2);

if E2>E1
    mu1=mu1+mu_inc;
else
    mu1=mu1-mu_dec;
end
if E2old>E2
    mu2=mu2+mu_inc;
else
    mu2=mu2-mu_dec;
end

% Wrt input weights

dPhijdVj=-network.a{1}.*exp(-nnoutput.vj{1});
dVjdWij=nnoutput.dvjwij;

J=[];
d=[];
for i=1:size(dVjdWij,2);
dEdWij=dVjdWij{1,i}.*dPhijdVj*ekq';
d=[d;LM(dEdWij,E2,mu2)];
end
d=d';

network.IW{1}=network.IW{1}+d;
E2old=E2;
end
function d=LM(J,E,mu)
d=((J'*J)+mu)^(-1)*J'*E;
end
