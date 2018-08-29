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

%% Split data into training set and validation set: 
fr_train=0.6;
fr_val=1-fr_train;
[X_train,X_val,Y_train,Y_val]=splitData([atrue,Btrue],Cm,fr_train,fr_val,1);

%% Linear Regression problem 
order=0; %polynomial order
errold=inf;
errnew=1e30;
%Increase order until error increases (due to floating point inaccuracies) 
while errnew<errold
errold=errnew;
[A,est]=OLSQ_est(order,X_train,Y_train); %use training set
estimatedCm=A*est;
resi=Y_train-estimatedCm;
errnew=sum(resi.^2);
order=order+1;
disp(order)
disp(errnew)
end
order=order-1;
err_train=errold;
[A,est]=OLSQ_est(order,X_train(:,1:2),Y_train);
estimatedCm_train=A*est;
res_train=Y_train-estimatedCm;
TRIeval = delaunayn([atrue Btrue]);

figure
plot3(X_train(:,1),X_train(:,2),estimatedCm_train,'.k');
grid();
hold on
trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');
legend('Linear regression model','full dataset');

%% Model-error based validation 
estimatedCm_val=calc_poly_output(est,X_val); %calculate output of polynomial with parameters found from training dataset
res_val=Y_val-estimatedCm_val;
err_val=sum(res_val.^2);
[err_autoCorr,lags]=xcorr(res_val-mean(res_val));
err_autoCorr=err_autoCorr/max(err_autoCorr);

figure
plot3(X_val(:,1),X_val(:,2),estimatedCm_val, '.k');
hold on
trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');
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
%%
function [A,est]=OLSQ_est(order,X,Y)
nrvars=size(X,2);
A=zeros(size(X,1),order+1);
A(:,1)=1;

for i=1:order 
    A(:,i+1)=(sum(X,2)).^i;
end

est=((A'*A)^(-1))*A'*Y;
end

function Y=calc_poly_output(est,X)
    n=length(est)-1; %polynomial order
    Y=est(1)*ones(size(X,1),1); %first element (x+y)^0 
    for i = 1:n
        Y=Y+est(i+1).*sum(X,2).^i; % Y=sum(ti*(x+y)^i
    end
end


% figure
% title('residual');
% trisurf(TRIeval,atrue,Btrue,resi,'EdgeColor','None');
% figure
% title('residual');
% plot(T,resi);
% hold on
% plot([0 T(length(T))],[meanres meanres], '--');
% legend('Cm error','mean error');
% pause

% %% Method 2 Binomial theorem 
% A2=ones(length(atrue),1);
% for n=1:order
%     term=zeros(length(atrue),1);
%     for k=0:n
%         coef=factorial(n)/(factorial(k)*(factorial(n-k)));
%         term=term+(coef*atrue.^k.*Btrue.^(n-k));
%     end
%     A2=[A2,term];
% end
% est2=((A2'*A2)^(-1))*A2'*Cm;
% estimatedCm2=A2*est2;
% resi2=Cm-estimatedCm2;
% quaderr2=sum(resi2.^2);
% 
% figure
% plot3(atrue,Btrue,Cm,'.k');
% grid();
% hold on
% trisurf(TRIeval,atrue,Btrue,estimatedCm2,'EdgeColor','None');
% title('method 2');
