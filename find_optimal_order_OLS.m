function [order, errl] = find_optimal_order_OLS(X_train,Y_train,X_val,Y_val,plotf,type)

order=0; %polynomial order
errold=inf;
errnew=1e30;
errl=[];
%Increase order until error increases (due to floating point inaccuracies) 
if plotf
figure
end
while errnew<errold
errold=errnew;
[A_train,theta]=OLSQ_est(order,X_train,Y_train,type); %use training set
[A_val,~]=OLSQ_est(order,X_val,Y_val,type); %use training set

estimatedCm=A_val*theta;
resi=Y_val-estimatedCm;
errnew=sum(resi.^2)/size(resi,1);
errl=[errl;errnew];
    if plotf
        cla
        plot3(X_train(:,1),X_train(:,2),estimatedCm,'.k');

        hold on
        trisurf(TRIeval,atrue,Btrue,Cm,'EdgeColor','None');

        refreshdata
        grid on;
        title(strcat('sumorder, MSE=',num2str(errnew)));
        legend('Linear regression model','full dataset');
        pause(0.1)
    end
    
    order=order+1;
    
end
order=order-2;

end