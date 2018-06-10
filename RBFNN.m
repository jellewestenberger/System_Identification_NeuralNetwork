clear all
close all
load('atrue.mat');
load('Btrue.mat');
load('Vtrue.mat');
% load('NetExampleRBF.mat');
load('T.mat');
load('F16traindata_CMabV_2018','Cm');
X=[atrue, Btrue, Vtrue]; %input vector 

Networktype='rbf'; %choose network type: radial basis function (rbf) or feedforward (ff)
nrInput=size(X,2);     %number of inputs being used

nrOutput=1; 
nrNodesHidden=[100]; %add columns to add more hidden layers;
nrHiddenlayers=size(nrNodesHidden,2);%number of nodes in the hidden layers
inputrange=[min(X); max(X)]';


NNset=createNNStructure(nrInput,nrHiddenlayers,nrNodesHidden,nrOutput,inputrange,Networktype);
  
%% ---CHECK---- %% 
 check=NNCheck(NNset,nrInput,nrNodesHidden,nrOutput);
%% ------------ %% 

X=X';

%% Linear regression 
k=1; %only valid for one hidden layer (for now);
A=zeros(size(X,2),size(NNset.IW{k},1));
for j=1:size(A,2)
   vk=0;
    for i=1:size(X,1)
    vk=vk+(X(i,:)-NNset.centers{k}(j,i)).^2*(NNset.IW{k}(j,i))^2;
    end
   A(:,j)=exp(-vk').*NNset.LW(j);
end
a_est=inv(A'*A)*A'*Cm; %least-squared estimators 
Cm_est=A*a_est;
NNset.a{k}=a_est;


%% Levenberg Marquard
%calculate dE/d Wjk   (Wjk = output weight);

% epoch = 3;
%  for it=1:epoch
%     disp(it);
[Y, hiddenoutput, Vj]=calcNNOutput(NNset,X); %nice for double checking with Cm_est (should be exactly equal)

ekq=Y'-Cm;


%% Before starting over
% disp(sum(ekq));
% k=1;
% dEdWjk{k}=zeros(size(NNset.IW{k},1),1); %output weights
% dEdWij{k}=zeros(size(NNset.IW{k}));  %input weights 
% dPhidVj{k}=zeros(size(NNset.IW{k},1),1);  %partial derivatives dphij/dvj
% for j=1:size(dEdWjk{k},1)
% yj=hiddenoutput{1}(j,:);    
% dEdWjk{k}(j)=sum(-1*ekq.*yj');
% dPhidVj{k}=-NNset.a{k}.*exp(-Vj{k});
% end
% for i=1:size(NNset.IW{k},2) 
%    s=X(i,:).*dPhidVj{k}.*NNset.LW';
%    s=-ekq'.*s;
%    dEdWij{k}(:,i)=sum(s,2);
% end
% 
% mu=1;
% for k = 1:size(NNset.IW,1)
%     IW=NNset.IW{k}'-((dEdWij{k}'*dEdWij{k}+mu*eye(size(dEdWij{k},2)))^(-1))*dEdWij{k}'*sum(ekq);
%     NNset.IW{k}=IW';
% end
% LW=NNset.LW-((dEdWjk{k}'*dEdWjk{k}+mu*eye(size(dEdWjk{k},2)))^(-1))*dEdWjk{k}'*sum(ekq);
% NNset.LW=LW;
%  end
%%


figure
plot3(atrue,Btrue,Y,'.b'); 
%% plotting 

% hold on
% plot3(atrue,Btrue,Cm_estimated,'.');
hold on
plot3(atrue,Btrue,Cm,'.k');
%%