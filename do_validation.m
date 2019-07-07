function do_validation(type,order,X_train,Y_train,X_val,Y_val,plotf,typenames)
global atrue Btrue Cm

%% Model-error based validation 
% Resiudal should be zero-mean white noise
% residuals should have constant variance and be uncorrelated

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

if strcmp(type,'simple')
    typename=typenames{1};
elseif strcmp(type,'sumorder')
    typename=typenames{2};
elseif strcmp(type,'allorder')
    typename=typenames{3};
end

if plotf
%     
% TRIeval = delaunayn([atrue Btrue]);   
% TRIeval2 = delaunayn(X_val);
% figure
% subplot(121)
% plot3(X_val(:,1),X_val(:,2),estimatedCm_val, '.k');
% hold on
% trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');
% legend(typename,'Full dataset')
% grid on 
% subplot(122)
% trisurf(TRIeval2,X_val(:,1),X_val(:,2),res_val,'EdgeColor','None');


figure
plot(res_val);
hold on 
plot([0,length(res_val)],[mean(res_val),mean(res_val)])
grid on 
xlim([0, size(res_val,1)])
title(typename,'interpreter','latex')%strcat('Residuals Values, order=',num2str(order)))
ylabel('Residual [-]')
xlabel('Sample');

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
hold on
text(-1500,-0.065,strcat(num2str(round(count_95*100,2)),'% of samples within 95% confidence'),'interpreter','latex')
pbaspect([2.5,1,1]);
ylim([-1.8*max(err_autoCorr(floor(size(err_autoCorr,1)/2)+2:end)),1.8*max(err_autoCorr(size(err_autoCorr,1)/2+1:end))])
xlabel('lags [#samples]')
ylabel('Auto-correlation [-]')
legend('Auto-Correlation','95% confidence interval')
title(typename,'interpreter','latex');
saveas(gcf,strcat('Report/plots/',type,'_rescorr.eps'),'epsc');
end



%% Statistical Based Validation
test=res_val'*res_val;
evar=(res_val'*res_val)/(size(res_val,1)-size(theta_train,1));
theta_cov=evar*(A_val'*A_val)^(-1);
theta_var=diag(theta_cov);

%other method
theta_cov2=pinv(A_val) * (res_val * res_val') * A_val * pinv(A_val) / A_val';
theta_var2=diag(theta_cov2);


fontsize=5;
figure;
subplot(121)
f=bar(theta_train,'Edgecolor','None');
xlim([1,size(theta_train,1)+1]);
ax=ancestor(f,'axes');
Yrule=ax.YAxis;
Xrule=ax.XAxis; 
Yrule.FontSize=fontsize;
Xrule.FontSize=fontsize;

grid on
ylabel('$\theta_i$ values','interpreter','latex','fontsize',fontsize);
xlabel('$i+j$','interpreter','latex','fontsize',fontsize)
if strcmp(type,'simple')
    xlabel('$i$','interpreter','latex','fontsize',fontsize)
end
title(typename,'interpreter','latex','fontsize',fontsize)
pbaspect([2.5 1 1])
subplot(122)
% bar(theta_var)
% hold on
f=bar(theta_var,'Edgecolor','None');
%set(gca,'Yscale','log')
xlabel('$i+j$','interpreter','latex','fontsize',fontsize)
if strcmp(type,'simple')
    xlabel('$i$','interpreter','latex','fontsize',fontsize)
end
ylabel('$\theta_i$ variance [-]','interpreter','latex','fontsize',fontsize);
title(typename,'interpreter','latex','fontsize',fontsize)
grid on
ax=ancestor(f,'axes');
Yrule=ax.YAxis;
Xrule=ax.XAxis; 
Yrule.FontSize=fontsize;
Xrule.FontSize=fontsize;
xlim([1,size(theta_train,1)+1]);
pbaspect([2.5 1 1])
saveas(gcf,strcat('Report/plots/',type,'estimator_vars.eps'),'epsc')

figure


end