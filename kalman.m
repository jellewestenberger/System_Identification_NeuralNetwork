clear all
close all
load_f16data2018
%% Test to see what happens if you reverse the measurements
% alpha_m=alpha_m(end:-1:1);
% Au=Au(end:-1:1);
% Aw=Aw(end:-1:1);
% beta_m=beta_m(end:-1:1);
% Cm=Cm(end:-1:1);
% 
% Vtot=Vtot(end:-1:1);
% Z_k=Z_k(end:-1:1,:);
% U_k=U_k(end:-1:1,:)*-1;



plotflag='y';
%% --------EXTENDED KALMAN FILTER-------
epsilon=1e-10;
maxIterations=100;
doIEKF=1;
dt=0.01;
N=size(U_k,1); % #measurements
D_x=U_k'; 
Z_k=Z_k';
D_x=[D_x;zeros(1,size(D_x,2))];
m=4; %state dimension
n=3; %output dimension
state_est_err=[];

x_0=[Vtot(1);0;0;0]; %initial state values

P_0=0.01*diag(ones(1,m)); %initial estimate of the covariance matrix

%process noise statistics
Ew= zeros(m,1); %there is no bias in the noise 
Q=diag([(1e-3)^2 (1e-3)^2 (1e-3)^2 0]);  %noise covariance 
w_k=Q*randn(m,N)+Ew.*ones(m,N); %noise signal 
D_x=D_x+w_k; % add process noise to system

%sensor noise statistics 
Ev=zeros(n,1);
R=diag([0.01^2 0.0058^2 0.112^2]);
v_k=R*randn(n,N)+Ev.*ones(n,N); %[va,vb,vv]

%noise input 
G= zeros(m); %noise input matrix  (no noise in accelerometer measurements) 
B=eye(m);  %input matrix 


XX_k1k1=zeros(m,N); %array for optimal predicted states 
z_pred=zeros(n,N);

%run extended kalman filters
disp('running kalman filter on measurements');
ti=0;
tf=dt;
x_k1k1=x_0;
z_k1k1=calc_MeasurementMat(x_k1k1,v_k(:,1));
XX_k1k1(:,1)=x_k1k1;
z_pred(:,1)=z_k1k1;
P_k1k1=P_0;

%% Check whether the Kalman filter will converge
check_observability

%% Start Kalman filter 
for k=2:N

%prediction x_k+1|k
[ti,x_kk_1]=ode45(@(t,x) calc_f(t,x,D_x(:,k-1)),[ti ti+tf],x_k1k1);
ti=ti(end);
x_kk_1=x_kk_1(end,:)';
% [ti,x_kk_1]=integrator(x_k1k1,D_x(:,k-1),ti,dt); %integrate the states (currently done in single steps using the perfect measurements)


%predicted output z_k+1|k

z_kk1=calc_MeasurementMat(x_kk_1,v_k(:,k));
%pertubation of state 
Fx=zeros(m,m); %The accelerometer values do not depend on the state values so Fx is zero.

[~,Psi]=c2d(Fx,B,dt);%
[Phi,Gamma]=c2d(Fx,G,dt);%

%prediction covariance matrix
P_kk_1=Phi*P_k1k1*Phi'+Gamma*Q*Gamma';

   if (doIEKF)

        % do the iterative part
        eta2    = x_kk_1;
        err     = 2*epsilon;

        itts    = 0;
        while (err > epsilon)
            if (itts >= maxIterations)
                fprintf('Terminating IEKF: exceeded max iterations (%d)\n', maxIterations);
                break
            end
            itts    = itts + 1;
            eta1    = eta2;

            % Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
            Hx       = calc_Jacob_out(eta1);
           
            
            % The innovation matrix
            Ve  = (Hx*P_kk_1*Hx' + R);

            % calculate the Kalman gain matrix
            K       = P_kk_1 * Hx' / Ve;
            % new observation state
            z_p     = calc_MeasurementMat(eta1,v_k(:,k)) ;%fpr_calcYm(eta1, u);

            eta2    = x_kk_1 + K * (Z_k(:,k) - z_p - Hx*(x_kk_1 - eta1));
            err     = norm((eta2 - eta1), inf) / norm(eta1, inf);
        end

        x_k1k1          = eta2;

   else
    
    %if not IEKF
    %pertubation of measurements
    Hx=calc_Jacob_out(x_kk_1);
  
    
    %kalman gain
    K=P_kk_1*Hx'*(Hx*P_kk_1*Hx'+R)^(-1);

    %optimal state
    x_k1k1=x_kk_1+K*(Z_k(:,k)-z_kk1);
   end

    XX_k1k1(:,k)=x_k1k1;
    %covariance correction
    P_k1k1=(eye(m)-K*Hx)*P_kk_1*(eye(m)-K*Hx)'+K*R*K';
    %corrected measurement
    z_k1k1=calc_MeasurementMat(x_k1k1,v_k(:,k));
    z_pred(:,k)=z_k1k1;
    state_est_err=[state_est_err, (x_k1k1-x_kk_1)];
end


%     
% atrue=(z_pred(1,:)-v_k(1,:))./(1+XX_k1k1(4,e));  
% Btrue=(z_pred(2,:)-v_k(2,:));
% Vtrue=(z_pred(3,:)-v_k(3,:));
atrue=(z_pred(1,:))./(1+XX_k1k1(4,end));  
atrue2=atan(XX_k1k1(3,:)./XX_k1k1(1,:));
figure
diff=atrue-atrue2;
plot(atrue)
hold on 
plot(atrue2)
legend('from upwash','from accelerometer')
grid()


% figure
% plot(atrue)
% hold on
% plot(atrue2)

Btrue=(z_pred(2,:));
Vtrue=(z_pred(3,:));
%NOT SURE IF NEED TO BE REMOVED FROM TRUE MEASUREMENTS OR PREDICTED
%MEASUREMENTS 
% atrue=(alpha_m'-v_k(1,:))./(1+XX_k1k1(4,:));
% Btrue=(beta_m'-v_k(2,:));
% Vtrue=(Vtot'-v_k(3,:));
% 

T=[0:dt:(10000*dt)];

if plotflag == 'y'
figure()
% subplot(211)

plot(T,z_pred(1,:));
hold on
plot(T,alpha_m);
hold on
plot(T,atrue);
hold on
plot(T,XX_k1k1(4,:));
pbaspect([2 1 1])
title('alpha');
grid()
l=legend('Approximated alpha','Measured alpha','reconstructed true alpha', 'estimated upwash coeficient', 'Location','northwest');
l.FontSize=8;
xlabel('time [s]')
ylabel('angle of attack [rad]')
saveas(gcf,'Report/plots/alpharecon.eps','epsc')
% subplot(212)
% plot(T,(z_pred(1,:)'-alpha_m))
% grid()

figure
plot(T,z_pred(2,:));
hold on
plot(T,beta_m);
% hold on
% plot(T,Btrue);
title('beta');
legend('predicted output','measured output','estimated true beta');
xlabel('time [s]')
ylabel('sideslip angle [rad]')


figure
plot(T,z_pred(3,:));
hold on
plot(T,Vtot);
plot(T,Vtrue);
title('V')
legend('predicted output','measured output','estimated true V');
xlabel('time [s]')
ylabel('velocity [m/s]')

figure
plot(T,(z_pred(1,:)'-alpha_m(:)));

TRIeval = delaunayn([atrue' Btrue']);

figure
plot3(atrue',Btrue',Cm,'.k');
xlabel('alpha')
ylabel('beta');
hold on
plot3(alpha_m',beta_m',Cm,'.b');
grid();
% hold on
% trisurf(TRIeval,atrue',Btrue',Cm,'EdgeColor','None');
figure
subplot(221)

plot(T(2:end),state_est_err(1,:))
title('u')
subplot(222)
plot(T(2:end),state_est_err(2,:))
title('v')
subplot(223)
plot(T(2:end),state_est_err(3,:))
title('w')
subplot(224)
plot(T(2:end),state_est_err(4,:))
title('Ca')

figure()
plot(T,XX_k1k1(4,:))
grid()
xlabel('Time [s]')
ylabel('C_{\alpha_{up}}[-]')
title('Estimated Upwash coefficient')
pbaspect([4 1 1])
ylim([min(XX_k1k1(4,:)) max(XX_k1k1(4,:))*1.1])
saveas(gcf,'Report/plots/caup.eps','epsc')

end

atrue=atrue';
Btrue=Btrue';
Vtrue=Vtrue';


save('atrue.mat','atrue');
save('Btrue.mat','Btrue');
save('Vtrue.mat','Vtrue');
save('T.mat','T');