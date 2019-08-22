clear all 
close all
a=0;
b=0;
c=1;
neval=3001;
%%
if a
a1=load('nnglobalsessionupto565.mat');
a2=load('nnglobalsession565to645.mat');

a3=load('nnglobalsession645845.mat');
a4=load('nnglobalsession865andup.mat');
a5=load('nnglobalsession1000.mat');
% a6=load('nnglobalsession2000.mat');
% a7=load('nnglobalsession1500.mat');
% a8=load('nnglobalsession12501750.mat
El=a1.El;
El=[El;a2.El(6:end,:);a3.El(6:end,:);a4.El;a5.El;];%a6.El;a7.El;a8.El];
El(:,2)=El(:,2)/neval;
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
saveas(gcf,'Report/plots/nnopti.eps','epsc');
end
%%
if b
b1=load("findoptimumRBFNN.mat");
b2=load("findoptimumRBFNNupto38.mat");

El2=b2.El; 
El3=b1.El;
El2=[El2;El3]./[1,neval];
Elm=[b2.El_mean;b1.El_mean]./[1,neval];
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
saveas(gcf,'Report/plots/nnopti2.eps','epsc');
end 
%%
if c
    clearvars -except c neval

c1=load('findoptimumfixevalupto650.mat');
c2=load('findoptimumfixeval660to.mat'); 
Elc=[c1.El;c2.El]./[1,neval];
Elmeanc=[c1.El_mean;c2.El_mean]./[1,neval];
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
xline(nsel,'--')
hold on 
plot(Elmeanc(j,1),Elmeanc(j,2),'.g','MarkerSize',20);
hold on 
text(1.05*nsel,0.6*minE,strcat(num2str(nsel)," neurons:"));
text(1.05*nsel,0.4*minE,strcat("Min MSE:",num2str(minE)));
grid on
xlim([min(Elc(:,1)),max(Elc(:,1))]);
xlabel("neurons");
ylabel("MSE [-]");
saveas(gcf,"Report/plots/findoptinn.eps",'epsc');
end
