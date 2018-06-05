clear all
close all
load('atrue.mat');
load('Btrue.mat');
load('Vtrue.mat');
load('NetExampleRBF.mat');
load('T.mat');
load('F16traindata_CMabV_2018','Cm');
X=[atrue, Btrue, Vtrue]; %input vector 

Networktype='rbf'; %choose network type: radial basis function (rbf) or feedforward (ff)
nrInput=size(X,2);     %number of inputs being used
nrHiddenlayers=1; 
nrOutput=1; 
nrNodesHidden=[10];    %number of nodes in the hidden layers
NNset.range=[min(X); max(X)]';

if strcmp(Networktype, 'ff')
    NNset.b{1,1}=zeros(nrNodesHidden,1); %input bias weights
    NNset.b{2,1}=zeros(nrOutput,1);     %output bias weights
    NNset.name{1,1}='feedforward';
elseif strcmp(Networktype,'rbf')
    NNset.trainFunct{1,1}='radbas';
    
    
    NNset.centers=zeros(nrNodesHidden,nrInput);     %center locations 
    NNset.center_dist='uniform';                     %center location distribution type
    if strcmp(NNset.center_dist,'uniform') %uniform center distribution (based on input range)
    for i=1:nrInput
       minin=NNset.range(i,1);
       minout=NNset.range(i,2);
       NNset.centers(:,i)=linspace(minin,minout,nrNodesHidden)';
    end
  
    end
%     if strcmp(NNset.center_dist,'meas_loc_weighted')
%         figure
%         h=histfit(atrue,2*nrNodesHidden);
%     end
    NNset.name{1,1}='rbf';
end


NNset.IW=ones(nrNodesHidden,nrInput); %1's for now  INPUT WEIGHTS
NNset.LW=-0.02*ones(nrOutput,nrNodesHidden(end));%OUTPUT WEIGHTS (end because only look at last hidden layer connects to output) 
if nrHiddenlayers>1
   NNset.HW 
end
NNset.trainParam.epochs=100;
NNset.trainParam.goal=0;
NNset.trainParam.min_grad=1e-10;
NNset.trainParam.mu=1e-3; %learning rate 
NNset.trainParam.mu_dec=0.1; 
NNset.trainParam.mu_inc=10;
NNset.trainParam.mu_max=1e10; 
NNset.trainalg=('trainlm');

%---CHECK----
 check=NNCheck(NNset,nrInput,nrNodesHidden,nrOutput);
%------------

% Nin=size(X
%Creating output---
X=X';
Nin=size(X,1);
L_end=size(X,2);
Nhidden=size(NNset.centers,1);
V1 = zeros(Nhidden,L_end);

for i=1:Nin
xij=X(i,:).*ones(size(V1));
% disp(NNset.centers(:,i));
cij=NNset.centers(:,i)*ones(1,L_end);
wj=NNset.IW(:,i);
V1=V1+(wj.*(xij-cij)).^2;
end;
%output for hidden layer
Y1=exp(-V1);
%output of output layer
Y2=NNset.LW*Y1;
figure
plot3(atrue,Btrue,Y2','.');
hold on
plot3(atrue,Btrue,Cm,'.k');
% hold on
% plot3(atrue,Btrue,Y2'-Cm,'.b');