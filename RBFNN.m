%% Load Data
clear all
close all
load('atrue.mat');
load('Btrue.mat');
load('Vtrue.mat');
load('T.mat');
% load('F16traindata_CMabV_2018','Cm');
load_f16data2018;

% load_f16data2018
% Cm=-1*Cm;
X=[alpha_m';beta_m'];
% X=[atrue'; Btrue'];%]; %input vector 
% X=[alpha_m,beta_m];

%% Create Initial Neural Network Structure
Networktype='rbf';      %choose network type: radial basis function (rbf) or feedforward (ff)
nrInput=size(X,1);      %number of inputs being used
nrOutput=1;             %Number of outputs
nrNodesHidden=[130] ;   %add columns to add more hidden layers;
X=X';
inputrange=[min(X); max(X)]'; 
X=X';   



%%---CHECK---- %% 
%  check=NNCheck(NNset,nrInput,nrNodesHidden,nrOutput);



%% Linear regression 
NNset_lin=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,1000,'random');
swit=1;
if swit
k=1; %only valid for one hidden layer (for now);
A=zeros(size(X,2),size(NNset_lin.IW{k},1));
for j=1:size(A,2)
   vk=0;
    for i=1:size(X,1)
    vk=vk+(X(i,:)-NNset_lin.centers{k}(j,i)).^2*(NNset_lin.IW{k}(j,i))^2;
    end
   A(:,j)=exp(-vk').*NNset_lin.LW(j);
end
a_est=inv(A'*A)*A'*Cm; %least-squared estimators 
Cm_est=A*a_est;
NNset_lin.a{k}=a_est; 
end
result=calcNNOutput(NNset_lin,X);

TRIeval = delaunayn(X(1:2,:)');

figure
trisurf(TRIeval,X(1,:)',X(2,:)',Cm,'edgecolor','none');
hold on
plot3(X(1,:),X(2,:),result.yk,'.')

%% Levenberg Marquard
NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,200,'ones');

 [NNset, ~]=trainNetwork(NNset,Cm,X,1,{'wo','wi','a','c'});
 result=calcNNOutput(NNset,X);

%% golden ratio search:
GR=(1+sqrt(5))/2.; 
a=100;
b=700;
c=b-((b-a)/GR);
d=a+((b-a)/GR);
c=floor(c); %we need integers for number of neurons
d=floor(d);
El=[0,0];

while abs(c-d)>=1
    if size(find(El(:,1)==c),1)==1    
        i=find(El(:,1)==c);
        E_c=El(i,2);
    else
        NN_c=createNNStructure(nrInput,[floor(c)],nrOutput,inputrange,Networktype,100,'ones');  
        [~,E_c]=trainNetwork(NN_c,Cm,X,1,{'wo','wi','a','c'});
        El=[El; c,E_c];
    end   
    if size(find(El(:,1)==d),1)==1    
        i=find(El(:,1)==d);
        E_d=El(i,2);
    else
        NN_d=createNNStructure(nrInput,[floor(d)],nrOutput,inputrange,Networktype,100,'ones');  
        [~,E_d]=trainNetwork(NN_d,Cm,X,1,{'wo','wi','a','c'});
        El=[El; d,E_d];
    end   

% NN_d=createNNStructure(nrInput,[d],nrOutput,inputrange,Networktype,'ones');   
% 
% [~,E_d]=trainNetwork(NN_d,Cm,X,10,0.1,100,1,[1,1,1,1]);
    if E_c< E_d
        b=d;
    else
        a=c;
    end
    c=round(b-((b-a)/GR),0);
    d=round(a+((b-a)/GR),0);

end
if size(find(El(:,1)==c),1)==1    
        i=find(El(:,1)==c);
        E_c=El(i,2);
    else
        NN_c=createNNStructure(nrInput,[floor(c)],nrOutput,inputrange,Networktype,100,'ones');  
        [~,E_c]=trainNetwork(NN_c,Cm,X,1,[1,1,1,1]);
        El=[El; c,E_c];
        
end
%%
TRIeval = delaunayn(X');

figure
trisurf(TRIeval,X(1,:)',X(2,:)',Cm,'edgecolor','none');
hold on
plot3(X(1,:),X(2,:),result.yk,'.')


