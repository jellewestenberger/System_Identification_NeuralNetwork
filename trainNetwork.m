function [NNsetmin, minerror,El,evl]=trainNetwork(NNset,Y_train,X_train,X_val,Y_val,plotf,selector,optimizeorder)

global eval 
eval=0;

eta=0.0000001;
disp('Number of Neurons:');
nn=length(NNset.LW);
disp(nn);
if size(X_train,1)==2
TRIeval = delaunayn(X_train');
TRIeval_val= delaunay(X_val');
end
eval_par=length(selector); %number of parameters being evaluated 
minerror=inf;
E={};
mu={};
stopcount=0;
for i=1:size(selector,2) 
    if strcmp(selector(i),'wi')||strcmp(selector{i},'c')
        mu{i}=mat2cell(ones(size(NNset.IW{1},2),1)*NNset.trainParam.mu,[1,1]);
        E{i}=mat2cell(ones(size(NNset.IW{1},2),1)*inf,[1,1]);
    else
        mu{i}={NNset.trainParam.mu};
        E{i}={inf};

    end    
    
end

mu_inc=NNset.trainParam.mu_inc;
mu_dec=NNset.trainParam.mu_dec;
mu_max=NNset.trainParam.mu_max;
trainalg=NNset.trainalg;
output=calcNNOutput(NNset,X_train); 
output_val=calcNNOutput(NNset,X_val); 
ekq=Y_train'-output.yk;
ekq_val=Y_val'-output_val.yk;
El=[sum(0.5*ekq_val.^2)];
Ek_val_old=1e9;
El2=[];
cyclel=[];
dEl=[]; 
evl=[eval];
cycle=1;
evaltot=NNset.trainParam.epochs;
dE=ones(size(selector))*-1e-9;
while eval<evaltot 
    cyclel=[cyclel, cycle];   
    NNset_old=NNset;
    ekq=Y_train'-output.yk;

    Ek=sum(0.5*ekq.^2);
    for p=1:length(selector)

        for j = 1:length(mu{1})            
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
                    output_val2=calcNNOutput(NNset,X_val);
                    Ek_val2=sum(0.5*(Y_val'-output_val2.yk).^2);
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
        selector=circshift(selector,shiftind,2);    
        mu=circshift(mu,shiftind,2)   ;
        E=circshift(E,shiftind,2);
        dE=circshift(dE,shiftind,2);
    end
    cycle=cycle+1;
    
    if El(end)<minerror
        minerror=El(end);
        NNsetmin=NNset;       
    end
    
    n=size(X_val,2);    
    fprintf('[%i neurons, eval %i/%i] min MSE: %e, MSE gradient: %e \n',nn,eval,evaltot,2*minerror/n,min(dE)/n) % MSE is defined as sum(e^2)/n while the error above is calculated as sum(0.5*e^2)
    dEl=[dEl,min(dE)/n]; %save MSE gradient

    if cycle>5
        tr=dEl(size(dEl,2)-4:end);
        if sum(tr>-NNset.trainParam.min_grad)>=5 %if the total MSE gradient of one cycle is smaller than the minimum allowed gradient add 1 to the stop counter
            stopcount=1;
        else
            stopcount=0;
        end
    end

    if stopcount
        fprintf('Training stopped because the error has not consistently decreased with more than the minimum required gradient\n');
        break   
    end
    if eval>=evaltot
        fprintf("Total evaluations exceeded max allowed evaluations. Training Stopped\n");
    end


end



%% Functions 
function d=LM(J,E,mu)
d=((J'*J)+mu)^(-1)*J'*E;
d=d';
end

% Partial derivatives: Note that some partial derivatives have already been
% calculated in calcNNOuput and are stored in outputs
function dEdWjk=outputWeight(outputs,ekq)
                          
        dEdWjk=outputs.yi{1,2}*ekq'*(-1);

end

function dEdWij=radbasInputWeight(outputs,ekq,v)

    dPhijdVj=outputs.dphidvj{1};
    dVjdWij=outputs.dvjdwij;

    dEdWij=dVjdWij{1,v}.*dPhijdVj*ekq'*(-1).*NNset.LW';

end

    function dEdWij=tansigInputWeight(outputs,ekq,v)

        dPhijdVj=outputs.dphidvj{1};
        dVjdWij=outputs.dvjdwij;
        dEdWij=dVjdWij{1,v}.*dPhijdVj.*ekq*(-1).*NNset.LW';
        dEdWij=sum(dEdWij,2);

    end

    function dEda=radbasAmplitude(outputs,ekq)

        dEda=(-1)*exp(-outputs.vj{1})*ekq'.*NNset.LW';

    end

    function dEdCij=radbasCenter(outputs,ekq,i)

        dPhijdVj=-NNset.a{1}.*exp(-outputs.vj{1});
        dVjdCij=outputs.dvjcij;
        dEdCij=dVjdCij{1,i}.*dPhijdVj*ekq'*(-1).*NNset.LW';

    end

    function dEdbi= inputBias(outputs,ekq)


    dEdyj=(-1)*ekq.*NNset.LW';
    dEdbi=sum(dEdyj.*outputs.dphidvj{1},2);

    end
    function dEdbo=outputBias(ekq)
        dEdbo=sum((-1)*ekq);

    end


    function plotfig(output,scroll)
          
        if plotf
            if not(ishandle(2))
                figure(2)
            end
            clf(2)
            if size(X_train,1)==2
       
                subplot(221)                        
                trisurf(TRIeval,X_train(1,:)',X_train(2,:)',Y_train,'edgecolor','none');

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
               
                
            else

            plot(output.yk)
            title(strcat('evaluation ',num2str(cycle)));
            hold on
            plot(Y_val)
            
            end

            
          
            refreshdata
            drawnow

        end
    end
end