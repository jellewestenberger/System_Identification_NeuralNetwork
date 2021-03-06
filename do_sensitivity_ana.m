
nrInput=size(NNset.IW{1,1},2);
nrNodesHidden=size(NNset.LW,2);
nrOutput=size(NNset.LW,1);
inputrange=NNset.range;
fprintf("Performing Sensitivity analysis for ")
if strcmp(NNset.name,'feedforward')
    Networktype='ff';
    fprintf("feedforward neural network ");
elseif strcmp(NNset.name,'radbas')
    Networktype='rbf';
    fprintf("radial basis function neural network ");
end
fprintf("(%s)\n",NNset.trainalg);
trainParam=NNset.trainParam;
trainalg=NNset.trainalg;
epochs=trainParam.epochs;

 % the sets correspond to the sets in tables of chapter 4
%set 1
rng(50); %fix random seed for fair comparison of sensitivity

El=[];
Eprop={};
for k =1:5
NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,epochs,'random');
NNset.LW=randn(size(NNset.LW))*0.01;
NNset.IW{1,1}=randn(size(NNset.IW{1,1}));
NNset.b{1,1}=randn(size(NNset.b{1,1}))*0.01;
NNset.b{2,1}=randn(size(NNset.b{2,1}))*0.01;
NNset.trainalg=trainalg; %gradient descent = error back propagation 
NNset.trainParam=trainParam;
[NNset,Ei,Elist,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)]; %convert quadratic error to MSE
Eprop{k,1}=Elist;
Eprop{k,2}=evl;
end
if savef
    save(strcat(Networktype,trainalg,'set1'),'El','Eprop','evl');
end



%set 2
Eprop={};
El=[];
for k =1:5
NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NNset.LW=randn(size(NNset.LW))*0.01;
NNset.IW{1,1}=5*randn(size(NNset.IW{1,1}));
NNset.b{1,1}=randn(size(NNset.b{1,1}))*0.01;
NNset.b{2,1}=randn(size(NNset.b{2,1}))*0.01;
NNset.trainalg=trainalg; %gradient descent = error back propagation 
NNset.trainParam=trainParam;
[NNset,Ei,Elist,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)]; %convert quadratic error to MSE
Eprop{k,1}=Elist;
Eprop{k,2}=evl;
end
if savef
save(strcat(Networktype,trainalg,'set2'),'El','Eprop','evl');
end

%set 3

Eprop={};
El=[];
for k =1:5
NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NNset.LW=randn(size(NNset.LW))*0.01;
NNset.IW{1,1}=0.2*randn(size(NNset.IW{1,1}));
NNset.b{1,1}=randn(size(NNset.b{1,1}))*0.01;
NNset.b{2,1}=randn(size(NNset.b{2,1}))*0.01;
NNset.trainalg=trainalg; %gradient descent = error back propagation 
NNset.trainParam=trainParam;
[NNset,Ei,Elist,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)]; %convert quadratic error to MSE
Eprop{k,1}=Elist;
Eprop{k,2}=evl;
end
if savef
    save(strcat(Networktype,trainalg,'set3'),'El','Eprop','evl');
end


%set 4

Eprop={};
El=[];
for k =1:5
NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NNset.LW=randn(size(NNset.LW))*0.05;
NNset.IW{1,1}=randn(size(NNset.IW{1,1}));
NNset.b{1,1}=randn(size(NNset.b{1,1}))*0.01;
NNset.b{2,1}=randn(size(NNset.b{2,1}))*0.01;
NNset.trainalg=trainalg; %gradient descent = error back propagation 
NNset.trainParam=trainParam;
[NNset,Ei,Elist,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)]; %convert quadratic error to MSE
Eprop{k,1}=Elist;
Eprop{k,2}=evl;
end
if savef
    save(strcat(Networktype,trainalg,'set4'),'El','Eprop','evl');
end



%set 5

Eprop={};
El=[];
for k =1:5
NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NNset.LW=randn(size(NNset.LW))*0.002;
NNset.IW{1,1}=randn(size(NNset.IW{1,1}));
NNset.b{1,1}=randn(size(NNset.b{1,1}))*0.01;
NNset.b{2,1}=randn(size(NNset.b{2,1}))*0.01;
NNset.trainalg=trainalg; %gradient descent = error back propagation 
NNset.trainParam=trainParam;
[NNset,Ei,Elist,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)]; %convert quadratic error to MSE
Eprop{k,1}=Elist;
Eprop{k,2}=evl;
end
if savef
    save(strcat(Networktype,trainalg,'set5'),'El','Eprop','evl');
end


%set 6

Eprop={};
El=[];
for k =1:5
NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
NNset.LW=randn(size(NNset.LW))*0.01;
NNset.IW{1,1}=randn(size(NNset.IW{1,1}));
NNset.b{1,1}=randn(size(NNset.b{1,1}))*0.05;
NNset.b{2,1}=randn(size(NNset.b{2,1}))*0.01;
NNset.trainalg=trainalg; %gradient descent = error back propagation 
NNset.trainParam=trainParam;
[NNset,Ei,Elist,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,[{'bo','wo','bi','wi'}],0);
El=[El;Ei./(0.5*size(X_val,2)),evl(end)]; %convert quadratic error to MSE
Eprop{k,1}=Elist;
Eprop{k,2}=evl;
end
if savef
    save(strcat(Networktype,trainalg,'set6'),'El','Eprop','evl');
end

%set 7

Eprop={};
El=[];
for k =1:5
    NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
    NNset.LW=randn(size(NNset.LW))*0.01;
    NNset.IW{1,1}=randn(size(NNset.IW{1,1}));
    NNset.b{1,1}=randn(size(NNset.b{1,1}))*0.002;
    NNset.b{2,1}=randn(size(NNset.b{2,1}))*0.01;
    NNset.trainalg=trainalg; %gradient descent = error back propagation 
    NNset.trainParam=trainParam;
    [NNset,Ei,Elist,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,[{'bo','wo','bi','wi'}],0);
    El=[El;Ei./(0.5*size(X_val,2)),evl(end)]; %convert quadratic error to MSE
    Eprop{k,1}=Elist;
    Eprop{k,2}=evl;
end
if savef
    save(strcat(Networktype,trainalg,'set7'),'El','Eprop','evl');
end


%set 8

Eprop={};
El=[];
for k =1:5
    NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
    NNset.LW=randn(size(NNset.LW))*0.01;
    NNset.IW{1,1}=randn(size(NNset.IW{1,1}));
    NNset.b{1,1}=randn(size(NNset.b{1,1}))*0.01;
    NNset.b{2,1}=randn(size(NNset.b{2,1}))*0.05;
    NNset.trainalg=trainalg; %gradient descent = error back propagation 
    NNset.trainParam=trainParam;
    save('NNFFset8','NNset');
    [NNset,Ei,Elist,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,[{'bo','wo','bi','wi'}],0);
    El=[El;Ei./(0.5*size(X_val,2)),evl(end)]; %convert quadratic error to MSE
    Eprop{k,1}=Elist;
    Eprop{k,2}=evl;                                     
end
if savef
    save(strcat(Networktype,trainalg,'set8'),'El','Eprop','evl');
end

%set 9

Eprop={};
El=[];
for k =1:5
    NNset=createNNStructure(nrInput,nrNodesHidden,nrOutput,inputrange,Networktype,4000,'random');
    NNset.LW=randn(size(NNset.LW))*0.01;
    NNset.IW{1,1}=randn(size(NNset.IW{1,1}));
    NNset.b{1,1}=randn(size(NNset.b{1,1}))*0.01;
    NNset.b{2,1}=randn(size(NNset.b{2,1}))*0.002;
    NNset.trainalg=trainalg; %gradient descent = error back propagation 
    NNset.trainParam=trainParam;
    [NNset,Ei,Elist,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,[{'bo','wo','bi','wi'}],0);
    El=[El;Ei./(0.5*size(X_val,2)),evl(end)]; %convert quadratic error to MSE
    Eprop{k,1}=Elist;
    Eprop{k,2}=evl;
end
if savef
    save(strcat(Networktype,trainalg,'set9'),'El','Eprop','evl');
end



%% plotting 

if plotf
    close all   
    set={};
    nrsets=9; 
    w=floor(sqrt(nrsets-1));
    colors{1}=[0,12,255]/255; %blue 
    colors{2}=[66,255,0]/255; %green 
    colors{3}=[255,216,0]/255; %yellow 
    colors{4}=[212,150,0]/255; %orange 
    colors{5}=[255,0,0]/255; %red 
    colors{6}=[176,6,255]/255; %purple 
    colors{7}=[0,0,0]/255; %black 
    colors{8}=[2,246,255]/255; %light blue 
    colors{9}=[16,147,6]/255; %dark green   
    legstr={};

    f=figure('Position',[10,10,1500,750]);
    q=1;
    for k=2:2:nrsets
        clear h;
        tr=[1,k,k+1];
        subplot(w,w,q)
        c=1;
        for l=tr
            set{l}=load(strcat('Data/',Networktype,trainalg,'set',num2str(l),".mat"));
            for j=1:size(set{l}.Eprop,1)
               hi=semilogy(set{l}.Eprop{j,2},set{l}.Eprop{j,1}./(0.5*3001),'color',colors{l},'LineWidth',1.1);
               hold on
            end
        legstr{c}=strcat("set"," ",num2str(l));   
        h(c)=hi(1);
        c=c+1;
        end

    legend(h,legstr)
    xlabel('evaluations')
    ylabel('MSE')
    grid on 
    ylim([1e-5,1]);
        q=q+1;
        xlim([0,epochs+10]);
    end

    if savef
        saveas(gcf,strcat('Report/plots/erorpropset',Networktype,trainalg,'.eps'),'epsc')
    end
end
