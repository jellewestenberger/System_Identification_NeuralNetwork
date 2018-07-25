function NNsetmin=LevMar(NNset,Cm,X,mu_inc,mu_dec,evaltot,plotf,selector)
close all
% atrue=X(1,:);
% Btrue=X(2,:);
% Vtrue=X(3,:);

%calculate dE/d Wjk   (Wjk = output weight);

if size(X,1)==2
TRIeval = delaunayn(X');
end
eval_par=find(selector); %number of parameters being evaluated 
minerror=inf;
E=inf(1,length(selector)); %list for storing errors of individual steps
E_old=inf(1,length(selector));
mu=ones(size(E));
outputs=calcNNOutput(NNset,X); 
ekq=Cm'-outputs.yk;
El=[sum(0.5*ekq.^2)];
figure
for eval=1:evaltot
disp(E(eval_par(end))) %display newest error 
clf
%         close all;
%     yk=outputs.yk; %total output of neural network
%     yi=outputs.yi; % inputs to input weights in each layer 
% 
    if plotf
        
        if size(X,1)==2
            set(gcf, 'WindowState','fullscreen')
            
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
        
%         plot3(atrue,Btrue,yk,'.b'); 
        plot(outputs.yk)
        title(strcat('evaluation ',num2str(eval)));
        hold on
%         plot3(atrue,Btrue,Cm,'.k');
        plot(Cm)
        end
%         legend('Approximation','True');
        refreshdata
        drawnow
%         pause
        
    end
% 
%     ekq=Cm'-yk;         %IO mapping errors
%     E1=sum(0.5*ekq.^2); %squared error
%     El=[El, E1];
% 
% %     if plotf
% %         figure
% %         plot(El)
% %         drawnow
% %     end
% 
%     disp(E1);
    
    %% calculate partial derivative wrt to output weights
    if selector(1)
        ekq=Cm'-outputs.yk;
        E(1)=sum(0.5*ekq.^2);
        E_old(1)=E(1);
                           
        dEdWjk=outputs.yi{1,2}*ekq'*(-1);
        J=dEdWjk;
        d=LM(J,E(1),mu(1));
        NNset.LW=NNset.LW-d;
        
        outputs=calcNNOutput(NNset,X);        
        ekq=Cm'-outputs.yk;
        E(1)=sum(0.5*ekq.^2);
        
        if E(1)>E_old(1)   
            mu(1)=mu(1)+mu_inc;
        else
            mu(1)=mu(1)-mu_dec;
        end
        
    end
  
    selector=circshift(selector,-1,2);    
    mu=circshift(mu,-1,2)   ;
    E=circshift(E,-1,2);
    E_old=circshift(E_old,-1,2);

    %% calculate partial derivatives wrt input weights
    if selector(1) %switch LM on input weights on or off
        ekq=Cm'-outputs.yk;
        E(1)=sum(0.5*ekq.^2);
        E_old(1)=E(1);


        dPhijdVj=-NNset.a{1}.*exp(-outputs.vj{1});
        dVjdWij=outputs.dvjwij;

        d=[];

        for i=1:size(dVjdWij,2)
            dEdWij=dVjdWij{1,i}.*dPhijdVj*ekq'*(-1).*NNset.LW';
            d=[d;LM(dEdWij,E(1),mu(1))];
        end
        d=d';
        NNset.IW{1}=NNset.IW{1}-d;        
      
        outputs=calcNNOutput(NNset,X);        
        ekq=Cm'-outputs.yk;
        E(1)=sum(0.5*ekq.^2);
        
        
        if E(1)>E_old(1)
            mu(1)=mu(1)+mu_inc;
        else
            mu(1)=mu(1)-mu_dec;
        end

    end
    selector=circshift(selector,-1,2);    
    mu=circshift(mu,-1,2)   ;
    E=circshift(E,-1,2);
    E_old=circshift(E_old,-1,2);

    
    %% Calculate amplitudes
    if selector(1) %switch LM on input weights on or off
        ekq=Cm'-outputs.yk;
        E(1)=sum(0.5*ekq.^2);
        E_old(1)=E(1);

        dEda=(-1)*exp(-outputs.vj{1})*ekq'.*NNset.LW';
        d=LM(dEda,E(1),mu(1));
        NNset.a{1}=NNset.a{1}-d';            
      
        outputs=calcNNOutput(NNset,X);        
        ekq=Cm'-outputs.yk;
        E(1)=sum(0.5*ekq.^2);
        
        
        if E(1)>E_old(1)
            mu(1)=mu(1)+mu_inc;
        else
            mu(1)=mu(1)-mu_dec;
        end

    end
    selector=circshift(selector,-1,2);    
    mu=circshift(mu,-1,2)   ;
    E=circshift(E,-1,2);
    E_old=circshift(E_old,-1,2);
    
   %% calculate partial derivatives wrt center locations
    if selector(1) %switch LM on input weights on or off
        ekq=Cm'-outputs.yk;
        E(1)=sum(0.5*ekq.^2);
        E_old(1)=E(1);


        dPhijdVj=-NNset.a{1}.*exp(-outputs.vj{1});
        dVjdCij=outputs.dvjcij;

        d=[];

        for i=1:size(dVjdCij,2)
            dEdCij=dVjdCij{1,i}.*dPhijdVj*ekq'*(-1).*NNset.LW';
            d=[d;LM(dEdCij,E(1),mu(1))];
        end
        d=d';
        NNset.centers{1}=NNset.centers{1}-d;        
      
        outputs=calcNNOutput(NNset,X);        
        ekq=Cm'-outputs.yk;
        E(1)=sum(0.5*ekq.^2);
        
        
        if E(1)>E_old(1)
            mu(1)=mu(1)+mu_inc;
        else
            mu(1)=mu(1)-mu_dec;
        end

    end
    selector=circshift(selector,-1,2);    
    mu=circshift(mu,-1,2)   ;
    E=circshift(E,-1,2);
    E_old=circshift(E_old,-1,2);  
    
    

    
El=[El,E(eval_par(end))];
if El(end)<minerror
    minerror=El(end);
    NNsetmin=NNset;
end
end
disp('min error:')
disp(minerror)

function d=LM(J,E,mu)
d=((J'*J)+mu)^(-1)*J'*E;
end

end