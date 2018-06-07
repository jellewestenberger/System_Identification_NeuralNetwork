clear all
close all
% kalman
% clearvars -except atrue Btrue Vtrue T
dataname = 'F16traindata_CMabV_2018';
disp('loading measurements');
load(dataname, 'Cm')
load('atrue.mat');
load('Btrue.mat');
load('Vtrue.mat');
load('T.mat');


vars=3; %number of variables (in this case alpha, beta and 
n=6; %polynomial order
% A=zeros(size(T,2),vars*n+1);
A=zeros(size(T,2),n+1);
A(:,1)=1;

for i=1:n 
    A(:,i+1)=(atrue(:)+Btrue(:)).^i;
    if i>1
        A(:,i+1)=A(:,i+1)+A(:,i);
    end
end


% for i=1:n
%    if vars==3
%    blck=[atrue.^i Btrue.^i Vtrue.^i];
%    end
%    if vars==2
%        blck=[atrue.^i Btrue.^i];
%    end
%    A(:,(2+(i-1)*vars):((2+(i-1)*vars)+vars-1))=blck;
% end

est=((A'*A)^(-1))*A'*Cm;
estimatedCm=A*est;
resi=Cm-estimatedCm;
meanres=mean(resi);
TRIeval = delaunayn([atrue Btrue]);

figure
plot3(atrue,Btrue,Cm,'.k');
grid();
hold on
trisurf(TRIeval,atrue,Btrue,estimatedCm,'EdgeColor','None');

figure
title('residual');
trisurf(TRIeval,atrue,Btrue,resi,'EdgeColor','None');
figure
title('residual');
plot(T,resi);
hold on
plot([0 T(length(T))],[meanres meanres], '--');
legend('Cm error','mean error');
