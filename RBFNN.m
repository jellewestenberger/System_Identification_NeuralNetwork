%% Load Data
clear all
close all
load('atrue.mat');
load('Btrue.mat');
load('Vtrue.mat');
load('T.mat');


% load('F16traindata_CMabV_2018','Cm');
load_f16data2018;
% ind=[1:10:size(Cm,1)];
% Cm=Cm(ind);
% atrue=atrue(ind);
% Btrue=Btrue(ind);
% load_f16data2018
% Cm=-1*Cm;
% X=[alpha_m';beta_m'];
% Cm=normalize(Cm)
atrue_nom=normalize(atrue,'zscore');
btrue_nom=normalize(Btrue,'zscore');   
fr_train=0.7;
fr_val=1-fr_train;
[X_train,X_val,Y_train,Y_val]=splitData([atrue_nom,btrue_nom],Cm,fr_train,fr_val,1);

% X_train=[atrue_nom'; btrue_nom'];%]; %input vector 
% Cm=normalize(Cm,'zscore');
% X=[atrue'; Btrue'];%]; %input vector 


% X=[alpha_m,beta_m];

%% Create Initial Neural Network Structure
Networktype='rbf';      %choose network type: radial basis function (rbf) or feedforward (ff)
nrInput=size(X_train,2);      %number of inputs being used
nrOutput=1;             %Number of outputs
nrNodesHidden=[110] ;   %add columns to add more hidden layers;
% X_train=X_train';
inputrange=[0.8*min(X_train); 1.2*max(X_train)]'; 
X_train=X_train';   
X_val=X_val';


%%---CHECK---- %% 
%  check=NNCheck(NNset,nrInput,nrNodesHidden,nrOutput);
figure('Position',[100,10,1000,700])
%% Linear regression 

NNset_lin=createNNStructure(nrInput,[105],nrOutput,inputrange,Networktype,1000,'random',Cm);
% NNset_lin.IW{1}=NNset_lin.IW{1}*1e-1;
swit=1;
if swit
k=1; %only valid for one hidden layer (for now);
A=zeros(size(X_train,2),size(NNset_lin.IW{k},1));
for j=1:size(A,2)
   vk=0;
    for i=1:size(X_train,1)
    vk=vk+(X_train(i,:)-NNset_lin.centers{k}(j,i)).^2*(NNset_lin.IW{k}(j,i))^2;
    end
   A(:,j)=exp(-vk').*NNset_lin.LW(j);
end
a_est=(A'*A)^(-1)*A'*Y_train; %least-squared estimators 
Y_train_est=A*a_est;
NNset_lin.a{k}=a_est; 
end
result=calcNNOutput(NNset_lin,X_train);
E=(1/size(Y_train,1))*sum((result.yk'-Y_train).^2);
TRIeval = delaunayn(X_train(1:2,:)',{'Qt','Qbb','Qc'});


clf
subplot(121)
trisurf(TRIeval,X_train(1,:)',X_train(2,:)',Y_train,'edgecolor','none');
hold on
plot3(X_train(1,:),X_train(2,:),Y_train_est,'.')
xlabel('\alpha normalized')
ylabel('\beta normalized')
zlabel('C_m [-]')
legend('C_m data','NN-output','location','best','interpreter','latex')
pbaspect([1,1,1])
view(135,20)
title(strcat(num2str(size(NNset_lin.LW,2))," Neurons, MSE: ",num2str(E)),'interpreter','latex');
set(gcf,'Renderer','OpenGL');
hold on;
light('Position',[0.5 .5 15],'Style','local');
camlight('headlight');
material([.3 .8 .9 25]);
shading interp;
lighting phong;
drawnow();

subplot(122)
trisurf(TRIeval,X_train(1,:)',X_train(2,:)',Y_train-result.yk','edgecolor','none')
title('Residual','interpreter','latex')
pbaspect([1,1,1])
view(135,20)

saveas(gcf,strcat('Report/plots/linearNN',num2str(size(NNset_lin.LW,2)),NNset_lin.init,'.eps'),'epsc')
saveas(gcf,strcat('Report/plots/linearNN',num2str(size(NNset_lin.LW,2)),NNset_lin.init,'.jpg'))

%% Levenberg Marquard
NNset=createNNStructure(nrInput,105,nrOutput,inputrange,Networktype,1000,'random',Y_train);
NNset.trainalg='trainlm';
NNset.trainParam.mu=100;
NNset.trainParam.mu_inc=10;
NNset.trainParam.mu_dec=0.1;
% Y_train_norm=normalize(Y_train,'zscore');

% [~, ~,E1,evl1]=trainNetwork(NNset,Y_train,X,1,{'wi','a','c','wo'},0);
[~, ~,E2,evl2]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,1,{'wo','c','a','wi'},0);
% [~, ~,E3,evl3]=trainNetwork(NNset,Y_train,X,1,{'wo','c','a','wi'},1);
%%
% figure 
% semilogy(evl1,E1)
% hold on 
% semilogy(evl2,E2)
% hold on
% semilogy(evl3,E3)
% hold off 
% grid on 
% title('Training order comparison')
% legend('wi-a-c-wo','wo-c-a-wi','optimize order','interpreter','latex')
% xlabel('evaluation','interpreter','latex')
% ylabel('error [0.5(.)^2]','interpreter','latex')
% saveas(gcf,'Report/plots/ordercomp.eps','epsc')
 

%% golden ratio search:GR=(1+sqrt(5))/2.; 
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
        NN_c=createNNStructure(nrInput,[floor(c)],nrOutput,inputrange,Networktype,200,'random',Y_train);  
        [~,E_c]=trainNetwork(NN_c,Y_train,X_train,1,{'wo','wi','a','c'});
        El=[El; c,E_c];
    end   
    if size(find(El(:,1)==d),1)==1    
        i=find(El(:,1)==d);
        E_d=El(i,2);
    else
        NN_d=createNNStructure(nrInput,[floor(d)],nrOutput,inputrange,Networktype,200,'random',Y_train);  
        [~,E_d]=trainNetwork(NN_d,Y_train,X_train,1,{'wo','wi','a','c'});
        El=[El; d,E_d];
    end   

% NN_d=createNNStructure(nrInput,[d],nrOutput,inputrange,Networktype,'ones');   
% 
% [~,E_d]=trainNetwork(NN_d,Y_train,X,10,0.1,100,1,[1,1,1,1]);
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
        NN_c=createNNStructure(nrInput,[floor(c)],nrOutput,inputrange,Networktype,200,'random',Y_train);  
        [~,E_c]=trainNetwork(NN_c,Y_train,X_train,1,[1,1,1,1]);
        El=[El; c,E_c];
        
end
%%
TRIeval = delaunayn(X_train');

figure
trisurf(TRIeval,X_train(1,:)',X_train(2,:)',Y_train,'edgecolor','none');
hold on
plot3(X_train(1,:),X_train(2,:),result.yk,'.')


