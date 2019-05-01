function [NNsetmin, minerror]=trainNetwork(NNset,Cm,X,plotf,selector)
close all
% atrue=X(1,:);
% Btrue=X(2,:);
% Vtrue=X(3,:);
global eval 
eval=0;
%calculate dE/d Wjk   (Wjk = output weight);
eta=0.0000001;
disp('Number of Neurons:');
disp(length(NNset.LW));
if size(X,1)==2
TRIeval = delaunayn(X');
end
eval_par=length(selector); %number of parameters being evaluated 
minerror=inf;
E={};
mu={};
for i=1:size(selector,2) 
    if strcmp(selector(i),'wi')||strcmp(selector{i},'c')
        mu{i}=ones(size(NNset.IW{1},2),1);
        E{i}=ones(size(NNset.IW{1},2),1)*inf;
    else
        mu{i}=1;
        E{i}=inf;
    end    
end
 %log performance per parameter
% mu=ones(size(E))*NNset.trainParam.mu;
mu_inc=NNset.trainParam.mu_inc;
mu_dec=NNset.trainParam.mu_dec;
mu_max=NNset.trainParam.mu_max;
output=calcNNOutput(NNset,X); 
ekq=Cm'-output.yk;
init=1;%set this to a larger value if  you want to keep updating each weight until the error is smaller before going to the next weight.
El=[sum(0.5*ekq.^2)];
cyclel=[];
figure
evaltot=NNset.trainParam.epochs;
dE=inf;
for cycle=1:evaltot
    % disp(E(eval_par(end))) %display newest error 
    cyclel=[cyclel, cycle];   
      
        NNset_old=NNset;
        output=calcNNOutput(NNset,X);
        ekq=Cm'-output.yk;
        Ek=sum(0.5*ekq.^2);
        for p=1:length(selector)
            for j = 1:length(mu{1})            
%             NNset_old=NNset; 
%             
%             
%             output=calcNNOutput(NNset,X);
%             ekq=Cm'-output.yk;
%             Ek=sum(0.5*ekq.^2);
            accept=0;
            m=1;
            while not(accept)
            if strcmp(selector{1},'wi')
               if strcmp(NNset.trainFunct(1),'radbas')
                   J=radbasInputWeight(output,ekq,j);
               elseif strcmp(NNset.trainFunct(1),'tansig')
                   J=tansigInputWeight(output,ekq,j);
               end
               d=LM(J,Ek,mu{1}(j));
               NNset.IW{1}(:,j)=NNset.IW{1}(:,j)-d;
               Curpar= NNset.IW{1}(:,j);
            elseif strcmp(selector{1},'wo')
                J=outputWeight(output,ekq); 
                d=LM(J,Ek,mu{1}(j));
                NNset.LW(1,:)=NNset.LW(1,:)-d';
                Curpar=NNset.LW(1,:);
            elseif strcmp(selector{1},'a')
                J=radbasAmplitude(output,ekq);
                d=LM(J,Ek,mu{1}(j));
                NNset.a{1}=NNset.a{1}-d; 
                Curpar=NNset.a{1};
            elseif strcmp(selector{1},'c')
                J=radbasCenter(output,ekq,j);
                d=LM(J,Ek,mu{1}(j));
                NNset.centers{1}(:,j)=NNset.centers{1}(:,j)-d;
                Curpar=NNset.centers{1}(:,j);
            elseif strcmp(selector{1},'bi')
                J=inputBias(output,ekq);
                d=LM(J,Ek,mu{1}(j));
                NNset.b{1}=NNset.b{1}-d;  
                Curpar=NNset.b{1};
            elseif strcmp(selector{1},'bo')
                J=outputBias(ekq);
                d=LM(J,Ek,mu{1}(j));
                NNset.b{2}=NNset.b{2}-d;  
                Curpar= NNset.b{2};
            end
            output1=calcNNOutput(NNset,X);
            ekq1=Cm'-output1.yk;
            Ek1=sum(0.5*ekq1.^2);          
            if (Ek1>Ek)
                if m<5
               m=m+1; 
               NNset=NNset_old;
               mu{1}(j)=mu{1}(j)*NNset.trainParam.mu_inc;
                else 
                    accept=1;                    
                end
            else
                accept=1;
                 mu{1}(j)=mu{1}(j)*NNset.trainParam.mu_dec;
            end
            
            plotfig(output)
            if accept
                NNset_old=NNset;
                output=output1;
                ekq=ekq1;
                Ek=Ek1;
                
                E{1}(j)=Ek;                 
            end
            
            end
            end
            selector=circshift(selector,-1,2);    
            mu=circshift(mu,-1,2)   ;
            E=circshift(E,-1,2);
            
%             E_old=circshift(E_old,-1,2);
        
        end

         
    El=[El,Ek];
    if El(end)<minerror
        minerror=El(end);
        NNsetmin=NNset;       
    end
disp(minerror)    
end

disp('min error:')
disp(minerror)



%% Functions 
function d=LM(J,E,mu)
d=((J'*J)+mu)^(-1)*J'*E;
d=d';
end


function dEdWjk=outputWeight(outputs,ekq)
%         ekq=Cm'-outputs.yk;
%         E(1)=sum(0.5*ekq.^2);
%         E_old(1)=E(1);
                           
        dEdWjk=outputs.yi{1,2}*ekq'*(-1);
%         J=dEdWjk;
%         if strcmp(NNset.trainalg,'trainlm')
%         d=LM(J,E(1),mu{1});
%          elseif strcmp(NNset.trainalg,'trainbp')
%            d=mu{1}*J; 
%            d=d';
%         end
%         NNset.LW=NNset.LW-d;
end

function dEdWij=radbasInputWeight(outputs,ekq,v)
%     ekq=Cm'-outputs.yk;
%     E(1)=sum(0.5*ekq.^2);
%     E_old(1)=E(1);
    dPhijdVj=outputs.dphidvj{1};%-NNset.a{1}.*exp(-outputs.vj{1});
    dVjdWij=outputs.dvjdwij;
%     d=[];
    dEdWij=dVjdWij{1,v}.*dPhijdVj*ekq'*(-1).*NNset.LW';
%     d=LM(dEdWij,E(1),mu{1}(v));
%     for i=1:size(dVjdWij,2)% loop over input weights belonging to alpha, beta (and V)
%         dEdWij=dVjdWij{1,i}.*dPhijdVj*ekq'*(-1).*NNset.LW';
%         if strcmp(NNset.trainalg,'trainlm')
%         d=[d;LM(dEdWij,E(1),mu{1}(i))]; %first column for alpha, second for beta
%              
%         end
%     end
%     d=d';
%     NNset.IW{1}(:,v)=NNset.IW{1}(:,v)-d;    
end

function dEdWij=tansigInputWeight(outputs,ekq,v)
%     ekq=Cm'-outputs.yk;
%     E(1)=sum(0.5*ekq.^2);
%     E_old(1)=E(1);
    dPhijdVj=outputs.dphidvj{1};%(4*exp(-2*outputs.vj{1}))./((1+exp(-2*outputs.vj{1})).^2);
    dVjdWij=outputs.dvjdwij;%yi{1};
    dEdWij=dVjdWij{1,v}.*dPhijdVj*ekq'*(-1).*NNset.LW';
%     d=[];
%     d=LM(dEdWij,E(1),mu{1}(v));
%     for i=1:size(dVjdWij,1)
%        dEdWij=sum(dPhijdVj.*dVjdWij(i,:).*ekq*(-1).*NNset.LW',2); 
%        if strcmp(NNset.trainalg,'trainlm')
%        d=[d;LM(dEdWij,E(1),mu{1})];
%        elseif strcmp(NNset.trainalg,'trainbp')
%            d=[d;(mu{1}*dEdWij)'];
%        end
%     end
%     d=d';
%     NNset.IW{1}(:,v)=NNset.IW{1}(:,v)-d;  
end

function dEda=radbasAmplitude(outputs,ekq)
%     ekq=Cm'-outputs.yk;
%     E(1)=sum(0.5*ekq.^2);
%     E_old(1)=E(1);

    dEda=(-1)*exp(-outputs.vj{1})*ekq'.*NNset.LW';
%     if strcmp(NNset.trainalg,'trainlm')
%     d=LM(dEda,E(1),mu{1});
%     end
%     NNset.a{1}=NNset.a{1}-d';
end

function dEdCij=radbasCenter(outputs,ekq,i)
%     ekq=Cm'-outputs.yk;
%     E(1)=sum(0.5*ekq.^2);
%     E_old(1)=E(1);


    dPhijdVj=-NNset.a{1}.*exp(-outputs.vj{1});
    dVjdCij=outputs.dvjcij;

    d=[];

%     for i=1:size(dVjdCij,2)
        dEdCij=dVjdCij{1,i}.*dPhijdVj*ekq'*(-1).*NNset.LW';
%         if strcmp(NNset.trainalg,'trainlm')
%         d=[d;LM(dEdCij,E(1),mu{1})];
%         end
%     end
%     d=d';
%     NNset.centers{1}=NNset.centers{1}-d;       
end

    function dEdbi= inputBias(outputs,ekq)
              
%     ekq=Cm'-outputs.yk;
%     E(1)=sum(0.5*ekq.^2);
%     E_old(1)=E(1);
    dEdyj=(-1)*ekq.*NNset.LW';
    dEdbi=sum(dEdyj.*outputs.dphidvj{1},2);
%         if strcmp(NNset.trainalg,'trainlm')
%         d=LM(dEdbi,E(1),mu{1});
%         end
%         NNset.b{1}=NNset.b{1}-d';
    end
    function dEdbo=outputBias(ekq)
         
%     ekq=Cm'-outputs.yk;
%     E(1)=sum(0.5*ekq.^2);
%     E_old(1)=E(1);
    dEdbo=sum((-1)*ekq);
%     if strcmp(NNset.trainalg,'trainlm')
%         d=LM(dEdbo,E(1),mu{1});
%     end
%     NNset.b{end}=NNset.b{end}-d;
    end
    

    function plotfig(output)
          clf
        if plotf

            if size(X,1)==2
    %             set(gcf, 'WindowState','fullscreen')            
                subplot(121)                        
                trisurf(TRIeval,X(1,:)',X(2,:)',Cm,'edgecolor','none')
%                 hold on
%                 plot3(X(1,:)',X(2,:)',Cm,'.b');
                hold on
                if accept
                    plot3(X(1,:),X(2,:),output.yk,'.g')
                else
                    plot3(X(1,:),X(2,:),output.yk,'.')
                end
                xlabel('alpha')
                ylabel('beta')
                title(strcat('evaluation:',{' '},num2str(eval),{' '},'optimizing ',{' '},selector{1},'(',num2str(j),')'))
                grid()          
                subplot(222)            
                semilogy(El)                  
                 title(strcat('error: ', num2str(El(end))))
                 xlabel('optimization cycle')
                 ylabel('error value');
                 grid();
                 subplot(224)
                 if accept
                    plot(Curpar,'.g')
                 else
                      plot(Curpar,'.b')
                 end
                 xlim([1,size(NNset.IW{1},1)]);
                title(selector{1})
                xlabel('Neuron')
                ylabel('Gain')
                grid();
            else

            plot(output.yk)
            title(strcat('evaluation ',num2str(cycle)));
            hold on
            plot(Cm)
            end
    %         legend('Approximation','True');
            refreshdata
            drawnow
    %         pause

        end
    end
end