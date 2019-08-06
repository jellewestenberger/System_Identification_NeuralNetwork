clear all 
close all
a1=load('nnglobalsessionupto565.mat');
a2=load('nnglobalsession565to645.mat');

a3=load('nnglobalsession645845.mat');
a4=load('nnglobalsession865andup.mat');
a5=load('nnglobalsession1000.mat');
neval=3001;
El=a1.El;
El=[El;a2.El(6:end,:);a3.El(6:end,:);a4.El;a5.El];
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

figure('Position',[10,10,1200,400]);
plot(El(:,1),El(:,2),'.')
hold on 
plot(Elmean(:,1),Elmean(:,2));
grid on
xlabel('Nr neurons')
ylabel('MSE')
legend('final MSE single run','Average final MSE');
saveas(gcf,'Report/plots/nnopti.eps','epsc');