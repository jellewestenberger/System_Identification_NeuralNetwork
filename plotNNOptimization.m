% this file plots the results of the number of neurons optimization 
% 

clear all 
close all
savefl=0;

neval=3001;
%% RBF Global performance plot (section 3.3)

a1=load('Data/nnglobalsessionupto565.mat');
a2=load('Data/nnglobalsession565to645.mat');

a3=load('Data/nnglobalsession645845.mat');
a4=load('Data/nnglobalsession865andup.mat');
a5=load('Data/nnglobalsession1000.mat');

El=a1.El;
El=[El;a2.El(6:end,:);a3.El(6:end,:);a4.El;a5.El;];
Elmean=[];

k=0;
n_prev=El(1,1);
m=0;
i=1;
variance =[];
while i<size(El,1)
   q=find(El(:,1)==El(i,1));
   r=El(q,:);
   Elmean=[Elmean; mean(r,1)];
   variance=[variance;El(i,1),var(r(:,2))];
   i=q(end)+1;
   dummy=2;
end

[~,k]=sort(Elmean(:,1));
Elmean=Elmean(k,:);

figure('Position',[10,10,1200,400]);
plot(El(:,1),El(:,2),'.')
hold on 
plot(Elmean(:,1),Elmean(:,2));
hold on 
plot(Elmean(:,1),Elmean(:,2),'.k','markersize',10);
grid on
xlabel('Nr neurons')
ylabel('MSE')
legend('final MSE single run','Average final MSE');
if savefl
saveas(gcf,'Report/plots/nnopti.eps','epsc');
end

%%

b1=load("Data/findoptimumRBFNN.mat");
b2=load("Data/findoptimumRBFNNupto38.mat");

El2=b2.El; 
El3=b1.El;
El2=[El2;El3]./[1,0.5*neval];
Elm=[b2.El_mean;b1.El_mean]./[1,0.5*neval];
grad=diff(Elm(:,2));
grad=[Elm(2:end,1),grad];
grad_m=movmean(grad(:,2),5);
threshold=-1e-6;
window=10;
for k=window:size(grad_m,1)
    grads=grad_m(k-(window+-1):k);
    ns=grad(k-(window+-1):k,1);
    T=grads>threshold;
    if sum(T)==window
        n=ns(end);
        break
    end
end


figure('Position',[10,10,1200,800])
subplot(211)
plot(El2(:,1),El2(:,2),'.')
hold on
plot(Elm(:,1),Elm(:,2))
grid on
legend('final MSE single run','Average final MSE');
subplot(212)
plot(grad(:,1),grad(:,2))
hold on
plot(grad(:,1),grad_m);
grid on 
legend('Error gradient','Moving Average Error gradient (10 samples)','location','southeast')

if savefl
saveas(gcf,'Report/plots/nnopti2.eps','epsc');
end


c1=load('Data/findoptimumfixevalupto650.mat');
c2=load('Data/findoptimumfixeval660to.mat'); 
Elc=[c1.El;c2.El]./[1,0.5*neval];
Elmeanc=[c1.El_mean;c2.El_mean]./[1,0.5*neval];
minE=inf; 
search=1;
k=1;
window=15;
while search && k<=size(Elmeanc,1)
    if Elmeanc(k,2)<minE
        minE=Elmeanc(k,2);
        j=k;
    end
    
    if (k-j)>=window
        search=0;
    end
    
    k=k+1; 
end
    
nsel=Elmeanc(j,1);


figure('Position',[10,10,1200,400])
plot(Elc(:,1),Elc(:,2),'.');
hold on 
plot(Elmeanc(:,1),Elmeanc(:,2));

hold on 
plot(Elmeanc(:,1),Elmeanc(:,2),'.k');
hold on 
line([nsel,nsel],get(gca,'ylim'))
hold on 
plot(Elmeanc(j,1),Elmeanc(j,2),'.g','MarkerSize',20);
hold on 
text(1.05*nsel,0.6*minE,strcat(num2str(nsel)," neurons:"));
text(1.05*nsel,0.4*minE,strcat("Min MSE:",num2str(minE)));
grid on
xlim([min(Elc(:,1)),max(Elc(:,1))]);
xlabel("neurons");
ylabel("MSE [-]");
if savefl
saveas(gcf,"Report/plots/findoptinn.eps",'epsc');
end




load('Data/NNsetf.mat')
load('Data/atrue.mat');
load('Data/Btrue.mat');
load('Data/Vtrue.mat');
load('Data/T.mat');
load_f16data2018;
atrue_nom=normalize(atrue,'zscore');
btrue_nom=normalize(Btrue,'zscore');   
X=[atrue_nom,btrue_nom]';
fr_train=0.7;
fr_val=1-fr_train;
[X_train,X_val,Y_train,Y_val]=splitData([atrue_nom,btrue_nom],Cm,fr_train,fr_val,1);
X_denom=[atrue,Btrue]'.*(180/pi);
TRIeval = delaunayn(X_denom');
nnoutput=calcNNOutput(NNsetf,X);
Cmnn=nnoutput.yk;
MSE=sum((Cmnn'-Cm).^2)/size(X_denom,2);
figure('Position',[10,10,1800,600]);
subplot(131)
trisurf(TRIeval,X_denom(1,:),X_denom(2,:),Cm,'edgecolor','none')
hold on 
plot3(X_denom(1,:),X_denom(2,:),Cmnn,'.k','MarkerSize',5)
pbaspect([1,1,1])
legend('Measured','RBFNN','location','best')
view(135,20)
set(gcf,'Renderer','OpenGL');
hold on;
light('Position',[0.5 .5 15],'Style','local');
camlight('headlight');
material([.3 .8 .9 25]);
shading interp;
lighting phong;
title('Measured')
xlabel('\alpha [deg]')
ylabel('\beta [deg]')
zlabel('Cm [-]')

subplot(132)
trisurf(TRIeval,X_denom(1,:),X_denom(2,:),Cmnn,'edgecolor','none')
pbaspect([1,1,1])
view(135,20)
set(gcf,'Renderer','OpenGL');
hold on;
light('Position',[0.5 .5 15],'Style','local');
camlight('headlight');
material([.3 .8 .9 25]);
shading interp;
lighting phong;
title('RBFNN Output')
xlabel('\alpha [deg]')
ylabel('\beta [deg]')
zlabel('Cm [-]')    

subplot(133)
trisurf(TRIeval,X_denom(1,:),X_denom(2,:),(Cmnn'-Cm).^2,'edgecolor','none')
% plot(Cmnn'-Cm);
pbaspect([1,1,1])
view(135,20)
set(gcf,'Renderer','OpenGL');
hold on;
light('Position',[0.5 .5 15],'Style','local');
camlight('headlight');
material([.3 .8 .9 25]);
shading interp;
lighting phong;
title('Quadratic Residual')
xlabel('\alpha [deg]')
ylabel('\beta [deg]')
zlabel('Cm [-]^2')    
if savefl
saveas(gcf,'Report/plots/finalrbfnn.eps','epsc')
end


