clear all
close all
% kalman
% clearvars -except atrue Btrue Vtrue T
dataname = 'F16traindata_CMabV_2018';
load(dataname);
disp('loading measurements');
load(dataname, 'Cm')
load('atrue.mat');
load('Btrue.mat');
load('Vtrue.mat');
load('T.mat');
atrue=Z_k(:,1);
Btrue=Z_k(:,2);
plotf=1;
set(0, 'DefaultAxesTickLabelInterpreter','latex')
    set(0, 'DefaultLegendInterpreter','latex')
if plotf
TRIeval = delaunayn([atrue Btrue]);
end

%%



%% Split data into training set and validation set: 
fr_train=0.9;
fr_val=1-fr_train;
[X_train,X_val,Y_train,Y_val]=splitData([atrue,Btrue],Cm,fr_train,fr_val,1);
X_train=[atrue Btrue];
Y_train=Cm;
%% Linear Regression problem [SIMPLE]
order=0; %polynomial order
errold=inf;
errnew=1e30;
errl1=[];
%Increase order until error increases (due to floating point inaccuracies) 
if plotf
    figure
end
while errnew<errold
errold=errnew;
[A,theta]=OLSQ_est(order,X_train,Y_train,'simple'); %use training set
estimatedCm=A*theta;
resi=Y_train-estimatedCm;
errnew=sum(resi.^2);
errl1=[errl1;errnew];
order=order+1;


    if plotf
        cla
        plot3(X_train(:,1),X_train(:,2),estimatedCm,'.k');

        hold on
        trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');

        refreshdata
        grid();
        title(strcat('Simple, E=',num2str(errnew)))
        legend('Linear regression model','full dataset');
        pause(0.1)
    end
end


%% Linear Regression problem [sumorder]
order=0; %polynomial order
errold=inf;
errnew=1e30;
errl2=[];
%Increase order until error increases (due to floating point inaccuracies) 
if plotf
figure
end
while errnew<errold
errold=errnew;
[A,theta]=OLSQ_est(order,X_train,Y_train,'sumorder'); %use training set
estimatedCm=A*theta;
resi=Y_train-estimatedCm;
errnew=sum(resi.^2);
errl2=[errl2;errnew];
    if plotf
        cla
        plot3(X_train(:,1),X_train(:,2),estimatedCm,'.k');

        hold on
        trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');

        refreshdata
        grid();
        title(strcat('sumorder, E=',num2str(errnew)));
        legend('Linear regression model','full dataset');
        pause(0.1)
    end
    order=order+1;
end
order=order-2;
err_train=errold;
[A,theta_train]=OLSQ_est(order,X_train(:,1:2),Y_train,'sumorder');
estimatedCm_train=A*theta_train;
res_train=Y_train-estimatedCm;
E_train=sum(res_train.^2);
 if plotf
        cla
        plot3(X_train(:,1),X_train(:,2),estimatedCm_train,'.k');

        hold on
        trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');
        
        grid();
        title(strcat('sumorder, E=',num2str(err_train)));
        legend('Linear regression model','full dataset'); 
        refreshdata
        
 end

 
 
%% Linear Regression problem [allorder]
order=0; %polynomial order
errold=inf;
errnew=1e30;
errl3=[];
%Increase order until error increases (due to floating point inaccuracies) 
if plotf
figure
end
while errnew<errold
errold=errnew;
[A,theta]=OLSQ_est(order,X_train,Y_train,'allorder'); %use training set
estimatedCm=A*theta;
resi=Y_train-estimatedCm;
errnew=sum(resi.^2);
errl3=[errl3;errnew];
    if plotf
        cla
        plot3(X_train(:,1),X_train(:,2),estimatedCm,'.k');

        hold on
        trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');

        refreshdata
        grid();
        title(strcat('allorder, E=',num2str(errnew)));
        legend('Linear regression model','full dataset');
        pause(0.1)
    end
    order=order+1;
end
order=order-2;
err_train=errold;
[A,theta_train3]=OLSQ_est(order,X_train(:,1:2),Y_train,'allorder');
estimatedCm_train=A*theta_train3;
res_train=(Y_train-estimatedCm);
E_train=sum(res_train.^2);
 if plotf
        cla
        plot3(X_train(:,1),X_train(:,2),estimatedCm_train,'.k');

        hold on
        trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');
        
        grid();
        title(strcat('allorder, E=',num2str(err_train)));
        legend('Linear regression model','full dataset'); 
        refreshdata
        
 end

%% Plot results of different polynomial types
figure()
plot([0:size(errl1,1)-1],errl1);
hold on
plot([0:size(errl2,1)-1],errl2);
hold on
plot([0:size(errl3,1)-1],errl3);
grid()
pbaspect([3 1 1])
xticks([0:1:max([size(errl3,1),size(errl2,1),size(errl1,1)])])
xlabel('Order') 
ylabel('Accuracy of fit [-]^2')
legend('$C_m=\sum_{i=0}^{n} \theta_i\left(\alpha+\beta\right)^i $','$C_m=\sum_{i+j=n}^n \theta_{i,j}\alpha^i\beta^j$','$C_m=\sum_{i,j}^n \theta_{i,j}\alpha^i\beta^j$','Interpreter','latex')
title('Influence of polynomial order on accuracy of fit');
saveas(gcf,'Report/plots/orderinfl.eps','epsc')
%% Model-error based validation 
[A_val,~]=OLSQ_est(order,X_val,Y_val,'sumorder');

estimatedCm_val=A_val*theta_train;
% estimatedCm_val=calc_poly_output(est,X_val); %calculate output of polynomial with parameters found from training dataset
res_val=Y_val-estimatedCm_val;
err_val=sum(res_val.^2);
[err_autoCorr,lags]=xcorr(res_val-mean(res_val));
err_autoCorr=err_autoCorr/max(err_autoCorr);

figure
plot3(X_val(:,1),X_val(:,2),estimatedCm_val, '.k');
hold on
trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');
legend('Linear regression validation','Full dataset')
grid()
figure
plot(res_val);
hold on 
plot([0,length(res_val)],[mean(res_val),mean(res_val)])
figure
plot(lags,err_autoCorr);
hold on
plot(lags([1,end]),[1.96/sqrt(length(res_val)),1.96/sqrt(length(res_val))],'--');
hold on
plot(lags([1,end]),[-1.96/sqrt(length(res_val)),-1.96/sqrt(length(res_val))],'--');




%% Functions
function [A,theta]=OLSQ_est(order,X,Y,type)
nrvars=size(X,2);

if strcmp(type,'simple')
A=zeros(size(X,1),order+1);
A(:,1)=1;

for i=1:order 
    A(:,i+1)=(sum(X,2)).^i;
end
elseif strcmp(type,'allorder')

    ordl=0:order;
    ordl=(ordl.*ones(size(ordl,2),nrvars)')';

    exps=ordl(1,:);
    %find all combinations of exponentials 
    while(sum(ordl(1,:)==order)<nrvars) 
       a=circshift(ordl,-1);
       if ordl(1,1)==order
           nr=sum(ordl(1,1:(end-1))==order);
           ordl(:,1:nr+1)=a(:,1:nr+1);
       else
           ordl(:,1)=a(:,1);
       end
       exps=[exps;ordl(1,:)];   
    end
    A=x2fx(X,exps);

d=2;
elseif strcmp(type,'sumorder')
    exps=zeros(1,nrvars);
    for k=1:order
    exps=[exps;exponentials(nrvars,k)];
    end
    A=x2fx(X,exps);
end
theta=((A'*A)^(-1))*A'*Y;
end

function Y=calc_poly_output(theta,X)
    n=length(theta)-1; %polynomial order
    Y=theta(1)*ones(size(X,1),1); %first element (x+y)^0 
    for i = 1:n
        Y=Y+theta(i+1).*sum(X,2).^i; % Y=sum(ti*(x+y)^i
    end
end
function exps=exponentials(vars,order)
if vars<=1
    exps=order;
    
else
    exps=zeros(0,size(vars,2));
    
    for i=order:-1:0
        rc=exponentials(vars-1,order-i);
        exps=[exps;i*ones(size(rc,1),1),rc];
    end
end
end

