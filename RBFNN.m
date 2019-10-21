clear all
close all
%% Settings
fprintf("Don't forget to set which parts of the report must be performed\n at the beginning of RBFNN.m\n");
do_lin_regress=1;           % perform linear regression (section 3.1.1)
do_compare_par_order=0;     % compare different order of parameter training (section 3.2.2) 
do_optimize_neuron=0;       % find optimal number of neurons (section 3.3)
do_train_optimal=0;         % train optimal network more extensively

plotf=1; %plot final results
plottraining=0; % plot during training. note that disabling plotting in trainNetwork.m speeds up training significantly 
savef=0; %save plots and results 


%% Load Data
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


%% Global Neural Network Parameters
Networktype='rbf';      %choose network type: radial basis function (rbf) or feedforward (ff)
nrInput=size(X_train,2);      %number of inputs being used
nrOutput=1;             %Number of outputs
nrNodesHidden=[110] ;   %add columns to add more hidden layers (training does not yet support multiple hidden layers);

inputrange=[0.8*min(X_train); 1.2*max(X_train)]'; % range of input values
X_train=X_train';   
X_val=X_val';

%% Linear regression 
if do_lin_regress
    n=103;
%     search=1; %search for best performing number of neurons (initial number must be close to optimal)
    E_old=inf;
    E=inf-1;
    result=0;
    while E<=E_old
        result_old=result;
        E_old=E;
        NNset_lin=createNNStructure(nrInput,[n],nrOutput,inputrange,Networktype,1000,'random');


        k=1; %only valid for one hidden layer
        A=zeros(size(X_train,2),size(NNset_lin.IW{k},1));
        for j=1:size(A,2)
           vk=0;
            for i=1:size(X_train,1)
            vk=vk+(X_train(i,:)-NNset_lin.centers{k}(j,i)).^2*(NNset_lin.IW{k}(j,i))^2;
            end
           A(:,j)=exp(-vk').*NNset_lin.LW(j);
        end
        a_est=(A'*A)^(-1)*A'*Y_train; %least-squared estimators 
        Y_train_est=A*a_est;
        NNset_lin.a{k}=a_est; 

        result=calcNNOutput(NNset_lin,X_val);
        E=(1/size(Y_val,1))*sum((result.yk'-Y_val).^2);
        NNset_lin_old=NNset_lin;
        n=n+1;
    end
    NNset_lin=NNset_lin_old;
    result=result_old;
    E=E_old;
    if plotf
        figure('Position',[100,10,1000,700])

        TRIeval = delaunayn(X(1:2,:)',{'Qt','Qbb','Qc'});
        TRIeval_val=delaunayn(X_val(1:2,:)',{'Qt','Qbb','Qc'});

        clf
        subplot(121)
        trisurf(TRIeval,X(1,:)',X(2,:)',Cm,'edgecolor','none');
        hold on
        plot3(X_val(1,:),X_val(2,:),result.yk,'.k')
        xlabel('\alpha normalized')
        ylabel('\beta normalized')
        zlabel('C_m [-]')
        legend({'Dataset','Output Model'},'Position',[-0.1,-0.1,1,1])
        pbaspect([1,1,1])
        view(135,20)
        title(strcat(num2str(size(NNset_lin.LW,2))," Neurons, MSE: ",num2str(E)));
        set(gcf,'Renderer','OpenGL');
        hold on;
        light('Position',[0.5 .5 15],'Style','local');
        camlight('headlight');
        material([.3 .8 .9 25]);
        shading interp;
        lighting phong;
        drawnow();

        subplot(122)
        trisurf(TRIeval_val,X_val(1,:)',X_val(2,:)',(Y_val-result.yk').^2,'edgecolor','none')
        title('Quadratic Residual')
        pbaspect([1,1,1])
        view(135,20)

        if savef
            saveas(gcf,strcat('Report/plots/linearNN',num2str(size(NNset_lin.LW,2)),NNset_lin.init,'.eps'),'epsc')
            saveas(gcf,strcat('Report/plots/linearNN',num2str(size(NNset_lin.LW,2)),NNset_lin.init,'.jpg'))
        end
    end
end
%% Levenberg Marquard
NNset=createNNStructure(nrInput,580,nrOutput,inputrange,Networktype,1000,'random');
NNset.trainalg='trainlm';
NNset.trainParam.mu=100;
NNset.trainParam.mu_inc=10;
NNset.trainParam.mu_dec=0.05;
NNset.trainParam.epochs=inf; 
NNset.trainParam.min_grad=1e-20;

%% Compare order of parameter training
if do_compare_par_order
    NNset=createNNStructure(nrInput,50,nrOutput,inputrange,Networktype,1000,'random');
    NNset.trainalg='trainlm';
    NNset.trainParam.mu=100;
    NNset.trainParam.mu_inc=10;
    NNset.trainParam.mu_dec=0.1;
    NNset.trainParam.min_grad=1e-50; 
    % Cm_norm=normalize(Cm,'zscore');
    neval=size(X_val,2);
    [~, ~,E1,evl1]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,{'wi','a','c','wo'},0);
    [~, ~,E2,evl2]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,{'wo','c','a','wi'},0);
    [~, ~,E3,evl3]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plottraining,{'wo','c','a','wi'},1);
    %
    E1=E1./(0.5*neval); %change error from sum(0.5*e^2) to MSE; 1/n*sum(e^2)
    E2=E2./(0.5*neval);
    E3=E3./(0.5*neval);
    %%
    if plotf
        figure('Position',[10,10,600,300])
        semilogy(evl1,E1)
        hold on 
        semilogy(evl2,E2)
        hold on
        semilogy(evl3,E3)
        hold off 
        grid on 
        title('Training order comparison')
        legend('wi-a-c-wo','wo-c-a-wi','optimize order','interpreter','latex')
        xlabel('Evaluation')
        ylabel('MSE [-]')
        xlim([0,1000])
        if savef
            saveas(gcf,'Report/plots/ordercomp.eps','epsc')
        end
    end
end
%% Search optimal number of neurons
if do_optimize_neuron
    counter=0;
    window=15;
    El=[];
    El_mean=[];
    minE=inf;
    nrit=2;
    search=1;
    n=10;
    if plotf
    figure()
    end
    while search
        Emean=0;
        for k=1:nrit
            NN_c=createNNStructure(nrInput,n,nrOutput,inputrange,Networktype,400,'random'); 
            NN_c.trainParam.min_grad=1e-15;
            [~,E_i]=trainNetwork(NN_c,Y_train,X_train,X_val,Y_val,plottraining,{'wo','c','a','wi'},0);
            Emean=Emean+(1/nrit)*E_i;
            El=[El;n,E_i];
        end
        El_mean=[El_mean;n,Emean];
        n=n+10;
        if Emean<minE
            minE=Emean;
            counter=0;
        else
            counter=counter+1;
        end
        if plotf
        cla();
        plot(El(:,1),El(:,2),'.')
        hold on
        plot(El_mean(:,1),El_mean(:,2))
        refreshdata()
        pause(0.01)
        xlabel('Number neurons')
        ylabel('quadratic error (0.5(.)^2)');
        if counter>=window
            search=0;
        end
        end
    end
end

%% Train network with optimal number of neurons (580) more extensively 
if do_train_optimal
Emin=inf;
El=[];
for k=1:5
        NN_c=createNNStructure(nrInput,580,nrOutput,inputrange,Networktype,inf,'random'); 
        NN_c.trainParam.min_grad=1e-13;
        [NNsetf,E_i,~,evl]=trainNetwork(NN_c,Y_train,X_train,X_val,Y_val,plottraining,{'wo','c','a','wi'},0);
        
        El=[El;n,E_i];
        if E_i<Emin
            Emin=E_i;
            save('NNsetf','NNsetf','evl')
        end   
end
end
