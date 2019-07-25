function [NNsetmin, minerror,El,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plotf,selector,optimizeorder)
% close all
% atrue=X(1,:);
% Btrue=X(2,:);
% Vtrue=X(3,:);
global eval 
eval=0;
plotf=1;

eta=0.0000001;
disp('Number of Neurons:');
disp(length(NNset.LW));
if size(X_train,1)==2
TRIeval = delaunayn(X_train');
TRIeval_val= delaunay(X_val');
end
eval_par=length(selector); %number of parameters being evaluated 
minerror=inf;
E={};
mu={};
for i=1:size(selector,2) 
    if strcmp(selector(i),'wi')||strcmp(selector{i},'c')
        mu{i}=mat2cell(ones(size(NNset.IW{1},2),1)*NNset.trainParam.mu,[1,1]);
        E{i}=mat2cell(ones(size(NNset.IW{1},2),1)*inf,[1,1]);
%         dE{i}=mat2cell(zeros(size(NNset.IW{1},2),1),[1,1]);
    else
        mu{i}={NNset.trainParam.mu};
        E{i}={inf};
%         dE{i}={0};
    end    
    
end
 %log performance per parameter
% mu=ones(size(E))*NNset.trainParam.mu;

mu_inc=NNset.trainParam.mu_inc;
mu_dec=NNset.trainParam.mu_dec;
mu_max=NNset.trainParam.mu_max;
trainalg=NNset.trainalg;
output=calcNNOutput(NNset,X_train); 
output_val=calcNNOutput(NNset,X_val); 
ekq=Y_train'-output.yk;
ekq_val=Y_val'-output_val.yk;
init=1;%set this to a larger value if  you want to keep updating each weight until the error is smaller before going to the next weight.
El=[sum(0.5*ekq_val.^2)];
Ek_val_old=0;
El2=[];
cyclel=[];
evl=[eval];
cycle=1;
evaltot=NNset.trainParam.epochs;
dE=ones(size(selector))*-1e-9;
while eval<evaltot
%     disp(num2str(cycle));
    % disp(E(eval_par(end))) %display newest error 
    cyclel=[cyclel, cycle];   
      
        NNset_old=NNset;
%         output=calcNNOutput(NNset,X);
        ekq=Y_train'-output.yk;
        
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
               if strcmp(trainalg,'trainlm')
               d=LM(J,Ek,mu{1}{j});
               elseif strcmp(trainalg,'traingd')
                   d=mu{1}{j}*J;
               end
               NNset.IW{1}(:,j)=NNset.IW{1}(:,j)-d;
               Curpar= NNset.IW{1}(:,j);
            elseif strcmp(selector{1},'wo')
                J=outputWeight(output,ekq); 
                if strcmp(trainalg,'trainlm')
                d=LM(J,Ek,mu{1}{j});
               elseif strcmp(trainalg,'traingd')
                   d=mu{1}{j}*J;
               end
                NNset.LW(1,:)=NNset.LW(1,:)-d';
                Curpar=NNset.LW(1,:);
            elseif strcmp(selector{1},'a')
                J=radbasAmplitude(output,ekq);
               if strcmp(trainalg,'trainlm')
               d=LM(J,Ek,mu{1}{j});
               elseif strcmp(trainalg,'traingd')
                   d=mu{1}{j}*J;
               end
                NNset.a{1}=NNset.a{1}-d; 
                Curpar=NNset.a{1};
            elseif strcmp(selector{1},'c')
                J=radbasCenter(output,ekq,j);
                if strcmp(trainalg,'trainlm')
               d=LM(J,Ek,mu{1}{j});
               elseif strcmp(trainalg,'traingd')
                   d=mu{1}{j}*J;
               end
                NNset.centers{1}(:,j)=NNset.centers{1}(:,j)-d;
                Curpar=NNset.centers{1}(:,j);
            elseif strcmp(selector{1},'bi')
                J=inputBias(output,ekq);
               if strcmp(trainalg,'trainlm')
               d=LM(J,Ek,mu{1}{j});
               elseif strcmp(trainalg,'traingd')
                   d=mu{1}{j}*J;
               end
                NNset.b{1}=NNset.b{1}-d;  
                Curpar=NNset.b{1};
            elseif strcmp(selector{1},'bo')
                J=outputBias(ekq);
                if strcmp(trainalg,'trainlm')
               d=LM(J,Ek,mu{1}{j});
               elseif strcmp(trainalg,'traingd')
                   d=mu{1}{j}*J;
               end
                NNset.b{2}=NNset.b{2}-d;  
                Curpar= NNset.b{2};
            end
            output_val=calcNNOutput(NNset,X_val);
            eval=eval+1;
            ekq_val=Y_val'-output_val.yk;
            Ek_val=sum(0.5*ekq_val.^2); 
            El2=[El2;Ek_val];
            if (Ek_val>Ek_val_old)
                if m<5
                   m=m+1; 
                   NNset=NNset_old;
                   if strcmp(trainalg,'trainlm') && mu{1}{j}<mu_max
                        mu{1}{j}=mu{1}{j}*NNset.trainParam.mu_inc;
                   elseif strcmp(trainalg,'traingd')
                        mu{1}{j}=mu{1}{j}*NNset.trainParam.mu_dec;
                   end
                else 
                    accept=1;                    
                end
            else
                accept=1;
                 if strcmp(trainalg,'trainlm')  
                 mu{1}{j}=mu{1}{j}*NNset.trainParam.mu_dec;
                 elseif strcmp(trainalg,'traingd') && mu{1}{j}<mu_max
                     mu{1}{j}=mu{1}{j}*NNset.trainParam.mu_inc;
                 end
            end
            
            plotfig(output_val,1)
            if accept
                NNset_old=NNset;
                output=calcNNOutput(NNset,X_train); 
                ekq=Y_train'-output.yk; 
                MSE_train=sum(ekq.^2)/size(ekq,2);
                MSE_val=sum(ekq_val.^2)/size(ekq_val,2);
                if (Ek_val-Ek_val_old)<0 && (Ek_val-Ek_val_old)<1e-2*min(dE)  %this prevents parameters never be chosen again if they increased the error for a run
                dE(1)=(Ek_val-Ek_val_old);
                dummy=3;
                else
                 dE(1)=1e-1*min(dE);
                end
                fprintf("MSE train: %f, MSE val: %f\n",MSE_train,MSE_val);
                
                E{1}{j}=Ek_val;   
                evl=[evl;eval];     
                El=[El,Ek_val];
                Ek_val_old=Ek_val;
                
            end
            
            end
            end
            if cycle>4 && optimizeorder
                [~,index]=min(dE);
                shiftind=-(index-1);
                dumm=2;
            else
                index=2;
                shiftind=-1;
            end
%             fprintf('next parameter: %s \n',selector{index})
            selector=circshift(selector,shiftind,2);    
            mu=circshift(mu,shiftind,2)   ;
            E=circshift(E,shiftind,2);
            dE=circshift(dE,shiftind,2);
            
            
%             E_old=circshift(E_old,-1,2);
        
        end
    cycle=cycle+1;
    
    if El(end)<minerror
        minerror=El(end);
        NNsetmin=NNset;       
    end
fprintf('\n min error: %f, gradient: %f \n',minerror,min(dE))
% disp(minerror)    
end

% disp('min error:')
% disp(minerror)



%% Functions 
function d=LM(J,E,mu)
d=((J'*J)+mu)^(-1)*J'*E;
d=d';
end

% Partial derivatives: Note that some partial derivatives have already been
% calculated in calcNNOuput and are stored in outputs
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
    

    function plotfig(output,scroll)
          
        if plotf
            if not(ishandle(2))
                figure(2)
            end
            clf(2)
            if size(X_train,1)==2
    %             set(gcf, 'WindowState','fullscreen')            
                subplot(221)                        
                trisurf(TRIeval,X_train(1,:)',X_train(2,:)',Y_train,'edgecolor','none');
%                 sp.XDataSource="X(1,:)'";
%                 sp.YDataSource="X(2,:)'";
%                 sp.ZDataSource="Cm";
%                 hold on
%                 plot3(X(1,:)',X(2,:)',Cm,'.b');
                hold on
                if accept
                    plot3(X_val(1,:),X_val(2,:),output_val.yk,'.g');
                else
                    plot3(X_val(1,:),X_val(2,:),output_val.yk,'.');
                end

                
                xlabel('alpha')
                ylabel('beta')
                title(strcat('evaluation:',{' '},num2str(eval),{' '},'optimizing ',{' '},selector{1},'(',num2str(j),'), MU: ',num2str(mu{1}{j})))
                grid on      
                view([1,1,1]);
                
                
                
                subplot(223)    
                QE=0.5*(output_val.yk-Y_val').^2;
                trisurf(TRIeval_val,X_val(1,:)',X_val(2,:)',QE,'edgecolor','none')
%                 plot3(X(1,:),X(2,:),QE,'.')
                
                xlabel('alpha')
                ylabel('beta')
                title('quadratic error')
                grid on      
                view([1,1,1]);
                
                if eval<=50 || not(scroll)
                subplot(224)            
                semilogy(evl,El,'Linewidth',2)          
                hold on
                semilogy(El2);
                hold on 
                semilogy(El2, '.k');
                 title(strcat('error: ', num2str(El(end))))
                 xlabel('Evaluation')
                 ylabel('error value');
                 grid on;
                 hold on
                else
                    subplot(224)
                    semilogy(evl(evl>(evl(end)-50)),El(evl>(evl(end)-50)),'Linewidth',2)
                    hold on 
                    semilogy(eval-50:eval,El2(end-50:end));
                    hold on 
                    semilogy(eval-50:eval,El2(end-50:end), '.k');
                    xlim([eval-50,eval]);
                    ylim([min(El(evl>(evl(end)-50))),max(El(evl>(evl(end)-50)))]);
                   
                    title(strcat('error: ', num2str(El2(end))))
                    xlabel('Evaluation')
                    ylabel('error value');
                    grid on;
                end
                
                
                subplot(222)
                semilogy(evl,El,'Linewidth',2)
                title(strcat('error: ', num2str(El(end))))
                xlabel('Evaluation')
                ylabel('error value');
                grid on
               
                
%                  subplot(224)
%                  if accept
%                     plot(Curpar,'.g')
%                  else
%                       plot(Curpar,'.b')
%                  end
%                  xlim([1,size(NNset.IW{1},1)]);
%                 title(selector{1})
%                 xlabel('Neuron')
%                 ylabel('Gain')
%                 grid on;
            else

            plot(output.yk)
            title(strcat('evaluation ',num2str(cycle)));
            hold on
            plot(Y_val)
            
            end
    %         legend('Approximation','True');
            
          
            refreshdata
            drawnow
    %         pause

        end
    end
end