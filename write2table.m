function write2table(thetas,orders,errors)

replaceLine = 1;
numLines = 5;
newText = 'This file originally contained a magic square';

fileID = fopen('Report\tableacc.tex','r');
mydata = cell(1, numLines);
for k = 1:numLines
   mydata{k} = fgetl(fileID);
end
fclose(fileID);


mydata{1}='\begin{table}[h!]';
mydata{2}=	'\centering';
mydata{3}=	'\begin{tabular}{|c|c|c|c|}';
mydata{4}= '\hline';
mydata{5}='type & number coefficients & minimal error & order \\ \hline \hline';
mydata{6}=strcat('\autoref{eq:polysimple}&',num2str(size(thetas.simple,1)),'&',num2str(errors(1)),'&',num2str(orders(1)),'\\ \hline');
mydata{7}=strcat('\autoref{eq:polyselect}&',num2str(size(thetas.sumorder,1)),'&',num2str(errors(2)),'&',num2str(orders(2)),'\\ \hline');
mydata{8}=strcat('\autoref{eq:polyallorder}&',num2str(size(thetas.allorder,1)),'&',num2str(errors(3)),'&',num2str(orders(3)),'\\ \hline');
mydata{9}='\end{tabular}';
mydata{10}='\caption{Accuracies of fit for the different polynomials}';
mydata{11}='\label{tab:accoffit}';
mydata{12}='\end{table}'; 

fileID = fopen('Report\tableacc.tex','w');
fprintf(fileID,'%s\n',mydata{:});
fclose(fileID);

end