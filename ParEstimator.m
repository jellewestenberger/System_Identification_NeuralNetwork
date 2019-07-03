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
plotf=1;  %plotflag
set(0, 'DefaultAxesTickLabelInterpreter','latex')
    set(0, 'DefaultLegendInterpreter','latex')
if plotf
TRIeval = delaunayn([atrue Btrue]);
end

%%



%% Split data into training set and validation set: 
fr_train=0.7;
fr_val=1-fr_train;
[X_train,X_val,Y_train,Y_val]=splitData([atrue,Btrue],Cm,fr_train,fr_val,1);
% X_train=[atrue Btrue];
% Y_train=Cm;
%% Linear Regression problem [SIMPLE]

[ordersimple,errl1]=find_optimal_order_OLS(X_train,Y_train,X_val,Y_val,0,'simple');
[A_simple,theta_simple]=OLSQ_est(ordersimple,X_train,Y_train,'simple');


%% Linear Regression problem [sumorder]

[ordersum,errl2]=find_optimal_order_OLS(X_train,Y_train,X_val,Y_val,0,'sumorder');
[A_sumorder,theta_sumorder]=OLSQ_est(ordersum,X_train(:,1:2),Y_train,'sumorder');
estimatedCm_train=A_sumorder*theta_sumorder;
res_train=Y_train-estimatedCm_train;
E_train=sum(res_train.^2)/size(res_train,1); %MSE
 if plotf
        cla
        plot3(X_train(:,1),X_train(:,2),estimatedCm_train,'.k');

        hold on
        trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');
        
        grid on;
        title(strcat('sumorder, MSE=',num2str(E_train)));
        legend('Linear regression model','full dataset'); 
        refreshdata
        
 end



 
%% Linear Regression problem [allorder]
[orderallorder,errl3]=find_optimal_order_OLS(X_train,Y_train,X_val,Y_val,0,'allorder');

[A_allorder,theta_allorder]=OLSQ_est(orderallorder,X_train(:,1:2),Y_train,'allorder');

estimatedCm_train=A_allorder*theta_allorder;
res_train=(Y_train-estimatedCm_train);
E_train=sum(res_train.^2)/size(res_train,1);
 if plotf
        cla
        plot3(X_train(:,1),X_train(:,2),estimatedCm_train,'.k');

        hold on
        trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');
        
        grid();
        title(strcat('allorder, MSE=',num2str(E_train)));
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
ylabel('MSE [-]')
legend('$C_m=\sum_{i=0}^{n} \theta_i\left(\alpha+\beta\right)^i $','$C_m=\sum_{i+j=n}^n \theta_{i,j}\alpha^i\beta^j$','$C_m=\sum_{i,j}^n \theta_{i,j}\alpha^i\beta^j$','Interpreter','latex')
title('Influence of polynomial order on accuracy of fit');
saveas(gcf,'Report/plots/orderinfl.eps','epsc')
thetatable.simple=theta_simple;
thetatable.sumorder=theta_sumorder;
thetatable.allorder=theta_allorder;

write2table(thetatable,[ordersimple,ordersum,orderallorder],[min(errl1),min(errl2),min(errl3)]); %write files to latex table for report


%% Model-error based validation 
% Resiudal should be zero-mean white noise
% residuals should have constant variance and be uncorrelated
type='allorder';
order=orderallorder;

[A_train,theta_train]=OLSQ_est(order,X_train,Y_train,type); %use training set
[A_val,~]=OLSQ_est(order,X_val,Y_val,type); %use training set

estimatedCm_val=A_val*theta_train;
% estimatedCm_val=calc_poly_output(est,X_val); %calculate output of polynomial with parameters found from training dataset
res_val=Y_val-estimatedCm_val;
err_val=sum(res_val.^2);
[err_autoCorr,lags]=xcorr(res_val-mean(res_val,1));
err_autoCorr = err_autoCorr/max(err_autoCorr);
conf_95 = 2/sqrt(size(err_autoCorr,1));
count_95 = size(find(abs(err_autoCorr)<conf_95),1)/size(err_autoCorr,1);
fileID = fopen(strcat('Report\',type,'_conf95.tex'),'w');
fprintf(fileID,'%s\n',strcat('$',num2str(round(count_95*100,1)),'\%$'));
fclose(fileID);

fprintf('%f percent lies within 95% confidence\n',count_95*100);


if plotf
TRIeval = delaunayn([atrue Btrue]);   

figure
plot3(X_val(:,1),X_val(:,2),estimatedCm_val, '.k');
hold on
trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');
legend('Linear regression validation','Full dataset')
grid on 

figure
plot(res_val);
hold on 
plot([0,length(res_val)],[mean(res_val),mean(res_val)])
grid on 
xlim([0, size(res_val,1)])
title(strcat('Residuals Values, order=',num2str(order)))
ylabel('\epsilon')
pbaspect([2.5,1,1])
legend('Model Residual',strcat('Mean residual =',num2str(mean(res_val))))
saveas(gcf,strcat('Report/plots/',type,'_resmean.eps'),'epsc');
figure
plot(lags,err_autoCorr);
hold on
plot(lags([1,end]),[conf_95,conf_95],'--k');
hold on
plot(lags([1,end]),[-conf_95,-conf_95],'--k');
grid on 
pbaspect([3,1,1]);
ylim([-1.3*max(err_autoCorr(floor(size(err_autoCorr,1)/2)+2:end)),1.3*max(err_autoCorr(size(err_autoCorr,1)/2+1:end))])
xlabel('lags [#samples]')
ylabel('Auto-correlation [-]')
legend('Auto-Correlation','95% confidence interval')
title(strcat('Residuals Normalized Correlation Values, order=',num2str(order)));
saveas(gcf,strcat('Report/plots/',type,'_rescorr.eps'),'epsc');
end

%% Statistical Based Validation
test=res_val'*res_val;
evar=(res_val'*res_val)/(size(res_val,1)-size(theta_train,1));
theta_cov=evar*(A_val'*A_val)^(-1);
theta_var=diag(theta_cov);

theta_cov2=pinv(A_val) * (res_val * res_val') * A_val * pinv(A_val) / A_val';
theta_var2=diag(theta_cov2);


figure
subplot(121)
bar(theta_train)
grid on
subplot(122)
grid on
% bar(theta_var)
% hold on
bar(theta_var2)
title(strcat('order=',num2str(order)))
grid on
saveas(gcf,strcat('Report/plots/',type,'estimator_vars.eps'),'epsc')

%% Functions


function Y=calc_poly_output(theta,X)
    n=length(theta)-1; %polynomial order
    Y=theta(1)*ones(size(X,1),1); %first element (x+y)^0 
    for i = 1:n
        Y=Y+theta(i+1).*sum(X,2).^i; % Y=sum(ti*(x+y)^i
    end
end


