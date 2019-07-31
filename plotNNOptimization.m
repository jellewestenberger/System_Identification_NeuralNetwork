load('nnglobalsession.mat')

close all
vars=[];
for j=1:5:(size(El,1))
    n=El(j:j+4,1);
    try
    variance=var(El(j:j+4,2));
    catch
        dummy=2
    end
    vars=[vars;n(1),variance];
end

figure('Position',[10,10,1000,600])
subplot(211)
plot(El(:,1),El(:,2),'.')
hold on 
plot(El_mean(:,1),El_mean(:,2))
grid on
legend('Quadratic error single run','Mean error')
xlabel('Number neurons')
ylabel('Final Quadratic Error (.)^2');
title('Error after 200 iterations');

subplot(212)
plot(vars(:,1),vars(:,2))
grid on 
xlabel('Number neurons')
ylabel('Variance')
title('Variance of errors');

saveas(gcf,'Report/plots/nnopti.eps','epsc')