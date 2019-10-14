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
X_nom=[atrue_nom,btrue_nom]';

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
if 0
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
end

%% Compare RBFNN FFNN and Polynomial 
if 0
FFNNset={};
FFEmin={};
FFEl={};
FFevl={};
FFNNsetmin={};

for it=1:5
FFNNset=createNNStructure(nrInput,100,nrOutput,inputrange,'ff',2000,'random');
FFNNset.LW=randn(size(FFNNset.LW))*0.01;
FFNNset.IW{1,1}=randn(size(FFNNset.IW{1,1}));
FFNNset.b{1,1}=randn(size(FFNNset.b{1,1}))*0.01;
FFNNset.b{2,1}=randn(size(FFNNset.b{2,1}))*0.05;
FFNNset.trainParam.min_grad=1e-13;
[FFNNsetmin{it},FFEmin{it},FFEl{it},FFevl{it}]=trainNetwork(FFNNset,Y_train,X_train,X_val,Y_val,0,{'bo','wo','bi','wi'},0);
end
save('FFComp','FFNNsetmin','FFEmin','FFEl','FFevl');





RBFFEmin={};
RBFFEl={};
RBFFevl={};
RBFFNNsetmin={};
for it=1:5
RBFNNset=createNNStructure(nrInput,100,nrOutput,inputrange,'rbf',2000,'random');
RBFNNset.LW=randn(size(RBFNNset.LW))*0.01;
RBFNNset.IW{1,1}=randn(size(RBFNNset.IW{1,1}));
RBFNNset.trainParam.min_grad=1e-13;
[RBFFNNsetmin{it},RBFFEmin{it},RBFFEl{it},RBFFevl{it}]=trainNetwork(RBFNNset,Y_train,X_train,X_val,Y_val,0,{'wo','c','a','wi'},0);
end
save('RBFComp','RBFFNNsetmin','RBFFEmin','RBFFEl','RBFFevl')

end


% plotting 
ffdat=load('FFComp.mat');
rbfdat=load('RBFComp.mat');

figure('Position',[10,10,1200,300])
neval=3001; %number of evaluation datapoints. For correcting quadratic error to MSE 
subplot(121)
for k=1:5
   semilogy(ffdat.FFevl{k},ffdat.FFEl{k}./(0.5*neval))
   hold on  
end
title("Feedforward Neural Network")
legend("Run 1","Run 2","Run 3","Run 4","Run 5")
xlim([0,2000])
ylim([1e-5,1e-2]);
grid on
hold off
xlabel('Evaluations')
ylabel('MSE [-]')
% pbaspect([2,1,1])
subplot(122)
for k=1:5
   semilogy(rbfdat.RBFFevl{k},rbfdat.RBFFEl{k}./(0.5*neval))
   hold on  
end
title("Radial Basis Function Neural Network")
xlim([0,2000])
ylim([1e-5,1e-2])
grid on
legend("Run 1","Run 2","Run 3","Run 4","Run 5")
hold off
xlabel('Evaluations')
ylabel('MSE [-]')
% pbaspect([2,1,1])

saveas(gcf,'Report/plots/convcomp.eps','epsc')
[~,ffbest]=min(cell2mat(ffdat.FFEmin));
[~,rbfbest]=min(cell2mat(rbfdat.RBFFEmin));

TRIeval = delaunayn(X_nom');

FFbestset=ffdat.FFNNsetmin{ffbest};
ffout=calcNNOutput(FFbestset,X_nom);

RBFbestset=rbfdat.RBFFNNsetmin{rbfbest};
rbfout=calcNNOutput(RBFbestset,X_nom);

poly=load('Data/sumorderpolyfull.mat');
polyout=poly.Afull*poly.theta_sumorder;

close all 

MSEff=sum((ffout.yk'-Cm).^2)./size(Cm,1);
MSErbf=sum((rbfout.yk'-Cm).^2)./size(Cm,1);
MSEpoly=sum((polyout-Cm).^2)/size(Cm,1);
figure('Position',[10,10,400,400])
% subplot(131)
trisurf(TRIeval,atrue.*180/pi,Btrue.*180/pi,Cm,'edgecolor','none')
hold on
plot3(atrue.*180/pi,Btrue.*180/pi,ffout.yk,'.k')
text(10,0,-0.12,strcat('MSE: ',num2str(MSEff)))
set(gcf,'Renderer','OpenGL');
view(135,20);
light('Position',[0.5 .5 15],'Style','local');
camlight('headlight');
material([.3 .8 .9 25]);
shading interp;
lighting phong 
pbaspect([1,1,1]);
title("Feedforward Neural Nework")
xlabel('alpha [deg]');
ylabel('beta [deg]');
zlabel('Cm [-]');
legend({'Dataset','Output Model'},'Position',[0.2,-0.2,1,1])
saveas(gcf,'Report/plots/comparisonff.eps','epsc')

figure('Position',[10,10,400,400])
% subplot(132)
trisurf(TRIeval,atrue.*180/pi,Btrue.*180/pi,Cm,'edgecolor','none')
hold on
plot3(atrue.*180/pi,Btrue.*180/pi,rbfout.yk,'.k')
text(10,0,-0.12,strcat('MSE: ',num2str(MSErbf)))
set(gcf,'Renderer','OpenGL');
view(135,20);
light('Position',[0.5 .5 15],'Style','local');
camlight('headlight');
material([.3 .8 .9 25]);
shading interp;
lighting phong 
pbaspect([1,1,1]);
title("Radial Basis Function Neural Network")
xlabel('alpha [deg]');
ylabel('beta [deg]');
zlabel('Cm [-]');
legend({'Dataset','Output Model'},'Position',[0.2,-0.2,1,1])
saveas(gcf,'Report/plots/comparisonrbf.eps','epsc')


figure('Position',[10,10,400,400])
% subplot(133)
trisurf(TRIeval,atrue.*180/pi,Btrue.*180/pi,Cm,'edgecolor','none')
hold on
plot3(atrue.*180/pi,Btrue.*180/pi,polyout,'.k')
text(10,0,-0.12,strcat('MSE: ',num2str(MSEpoly)))
set(gcf,'Renderer','OpenGL');
view(135,20);
light('Position',[0.5 .5 15],'Style','local');
camlight('headlight');
material([.3 .8 .9 25]);
shading interp;
lighting phong 
pbaspect([1,1,1]);
title("Polynomial Model")
xlabel('alpha [deg]');
ylabel('beta [deg]');
zlabel('Cm [-]');
legend({'Dataset','Output Model'},'Position',[0.2,-0.2,1,1])
saveas(gcf,'Report/plots/comparisonpoly.eps','epsc')


% saveas(gcf,'Report/plots/comparison.eps','epsc');





