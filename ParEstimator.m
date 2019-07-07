clear all
close all
global atrue Btrue Cm
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
typenames={'$C_m=\sum_{i=0}^{n} \theta_i\left(\alpha+\beta\right)^i $','$C_m=\sum_{i+j=n}^n \theta_{i,j}\alpha^i\beta^j$','$C_m=\sum_{i,j}^n \theta_{i,j}\alpha^i\beta^j$'};

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
legend(typenames{1},typenames{2},typenames{3},'Interpreter','latex')
title('Influence of polynomial order on accuracy of fit');
saveas(gcf,'Report/plots/orderinfl.eps','epsc')
thetatable.simple=theta_simple;
thetatable.sumorder=theta_sumorder;
thetatable.allorder=theta_allorder;

write2table(thetatable,[ordersimple,ordersum,orderallorder],[min(errl1),min(errl2),min(errl3)]); %write files to latex table for report


%% Validation 

do_validation('simple',orderallorder,X_train,Y_train,X_val,Y_val,1,typenames);
do_validation('sumorder',orderallorder,X_train,Y_train,X_val,Y_val,1,typenames);
do_validation('allorder',orderallorder,X_train,Y_train,X_val,Y_val,1,typenames);


