function [NNsetmin, minerror]=trainNetwork(NNset,Cm,X,plotf,selector)
close all
% atrue=X(1,:);
% Btrue=X(2,:);
% Vtrue=X(3,:);

%calculate dE/d Wjk   (Wjk = output weight);
eta=0.0000001;
disp('Number of Neurons:');
disp(length(NNset.LW));
if size(X,1)==2
TRIeval = delaunayn(X');
end
eval_par=length(selector); %number of parameters being evaluated 
minerror=inf;
E=inf(1,length(selector)); %list for storing errors of individual steps
E_old=inf(1,length(selector));
mu=ones(size(E))*NNset.trainParam.mu;
mu_inc=NNset.trainParam.mu_inc;
mu_dec=NNset.trainParam.mu_dec;
outputs=calcNNOutput(NNset,X); 
ekq=Cm'-outputs.yk;
El=[sum(0.5*ekq.^2)];
evall=[];
figure
evaltot=NNset.trainParam.epochs;
for eval=1:evaltot
    % disp(E(eval_par(end))) %display newest error 
    evall=[evall, eval];
    clf

        if plotf

            if size(X,1)==2
    %             set(gcf, 'WindowState','fullscreen')            
                subplot(121)                        
                trisurf(TRIeval,X(1,:)',X(2,:)',Cm,'edgecolor','none')
                hold on
                plot3(X(1,:),X(2,:),outputs.yk,'.')
                hold on
                subplot(122)            
                semilogy(El)            
                 title(strcat('error: ', num2str(El(end))))
                 xlabel('evaluation')
                 ylabel('error value');
                 grid();
            else

            plot(outputs.yk)
            title(strcat('evaluation ',num2str(eval)));
            hold on
            plot(Cm)
            end
    %         legend('Approximation','True');
            refreshdata
            drawnow
    %         pause

        end
        
        for p=1:length(selector)
            if strcmp(NNset.name{1},'rbf')
                if strcmp(selector(1),'wo')
                    outputWeight();
                elseif strcmp(selector(1),'wi')
                    if strcmp(NNset.trainFunct{1},'radbas')
                    radbasInputWeight();
                    end
                elseif strcmp(selector(1),'a')
                    radbasAmplitude()
                elseif strcmp(selector(1),'c')
                    radbasCenter();
                end
            end
            if strcmp(NNset.name{1},'feedforward')
                if strcmp(selector(1),'wo')
                    outputWeight();
                elseif strcmp(selector(1),'wi')
                    if strcmp(NNset.trainFunct{1},'tansig')
                    tansigInputWeight();
                    end
                elseif strcmp(selector(1),'bi')
                    inputBias();
                elseif strcmp(selector(1),'bo')
                    outputBias();
                    
                end
            end
            update();
            selector=circshift(selector,-1,2);    
            mu=circshift(mu,-1,2)   ;
            E=circshift(E,-1,2);
            E_old=circshift(E_old,-1,2);
        end

         
    El=[El,E(end)];
    if El(end)<minerror
        minerror=El(end);
        NNsetmin=NNset;
    end
end

disp('min error:')
disp(minerror)



%% Functions 
function d=LM(J,E,mu)
d=((J'*J)+mu)^(-1)*J'*E;
end
function update()
    outputs=calcNNOutput(NNset,X);  
    
    ekq=Cm'-outputs.yk;
    E(1)=sum(0.5*ekq.^2);

    if E(1)>E_old(1) 
        if mu(1)<NNset.trainParam.mu_max
        mu(1)=mu(1)+mu_inc;
        end
    else
        mu(1)=mu(1)-mu_dec;
    end
end

function outputWeight()
        ekq=Cm'-outputs.yk;
        E(1)=sum(0.5*ekq.^2);
        E_old(1)=E(1);
                           
        dEdWjk=outputs.yi{1,2}*ekq'*(-1);
        J=dEdWjk;
        if strcmp(NNset.trainalg,'trainlm')
        d=LM(J,E(1),mu(1));
         elseif strcmp(NNset.trainalg,'trainbp')
           d=mu(1)*J; 
           d=d';
        end
        NNset.LW=NNset.LW-d;
end

function radbasInputWeight()
    ekq=Cm'-outputs.yk;
    E(1)=sum(0.5*ekq.^2);
    E_old(1)=E(1);
    dPhijdVj=outputs.dphidvj{1};%-NNset.a{1}.*exp(-outputs.vj{1});
    dVjdWij=outputs.dvjdwij;
    d=[];

    for i=1:size(dVjdWij,2)% loop over input weights belonging to alpha, beta (and V)
        dEdWij=dVjdWij{1,i}.*dPhijdVj*ekq'*(-1).*NNset.LW';
        if strcmp(NNset.trainalg,'trainlm')
        d=[d;LM(dEdWij,E(1),mu(1))];
             
        end
    end
    d=d';
    NNset.IW{1}=NNset.IW{1}-d;    
end

function tansigInputWeight()
    ekq=Cm'-outputs.yk;
    E(1)=sum(0.5*ekq.^2);
    E_old(1)=E(1);
    dPhijdVj=outputs.dphidvj{1};%(4*exp(-2*outputs.vj{1}))./((1+exp(-2*outputs.vj{1})).^2);
    dVjdWij=outputs.dvjdwij{1};%yi{1};
    d=[];
    for i=1:size(dVjdWij,1)
       dEdWij=sum(dPhijdVj.*dVjdWij(i,:).*ekq*(-1).*NNset.LW',2); 
       if strcmp(NNset.trainalg,'trainlm')
       d=[d;LM(dEdWij,E(1),mu(1))];
       elseif strcmp(NNset.trainalg,'trainbp')
           d=[d;(mu(1)*dEdWij)'];
       end
    end
    d=d';
    NNset.IW{1}=NNset.IW{1}+d;  
end

function radbasAmplitude()
    ekq=Cm'-outputs.yk;
    E(1)=sum(0.5*ekq.^2);
    E_old(1)=E(1);

    dEda=(-1)*exp(-outputs.vj{1})*ekq'.*NNset.LW';
    if strcmp(NNset.trainalg,'trainlm')
    d=LM(dEda,E(1),mu(1));
    end
    NNset.a{1}=NNset.a{1}-d';
end

function radbasCenter()
    ekq=Cm'-outputs.yk;
    E(1)=sum(0.5*ekq.^2);
    E_old(1)=E(1);


    dPhijdVj=-NNset.a{1}.*exp(-outputs.vj{1});
    dVjdCij=outputs.dvjcij;

    d=[];

    for i=1:size(dVjdCij,2)
        dEdCij=dVjdCij{1,i}.*dPhijdVj*ekq'*(-1).*NNset.LW';
        if strcmp(NNset.trainalg,'trainlm')
        d=[d;LM(dEdCij,E(1),mu(1))];
        end
    end
    d=d';
    NNset.centers{1}=NNset.centers{1}-d;       
end

    function inputBias()
              
    ekq=Cm'-outputs.yk;
    E(1)=sum(0.5*ekq.^2);
    E_old(1)=E(1);
    dEdyj=(-1)*ekq.*NNset.LW';
    dEdbi=sum(dEdyj.*outputs.dphidvj{1},2);
        if strcmp(NNset.trainalg,'trainlm')
        d=LM(dEdbi,E(1),mu(1));
        end
        NNset.b{1}=NNset.b{1}-d';
    end
    function outputBias()
         
    ekq=Cm'-outputs.yk;
    E(1)=sum(0.5*ekq.^2);
    E_old(1)=E(1);
    dEdbo=sum((-1)*ekq)
    if strcmp(NNset.trainalg,'trainlm')
        d=LM(dEdbo,E(1),mu(1));
    end
    NNset.b{end}=NNset.b{end}-d
    end

end