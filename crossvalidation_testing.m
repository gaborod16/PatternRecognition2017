%% Cross validation test


dataX = ldab.X_train;
datay = ldab.y_train;
n_folds = 10;
n_data = numel(datay);
v_error = zeros(n_folds,1);

xval = ldab;
[itrn,itst] = crossval(n_data,n_folds);
final_matrix=zeros(2);

for i=1:n_folds
    xval.X_test = dataX(itst{1},:);
    xval.y_test = datay(:,itst{1});
    xval.X_train = dataX(itrn{1},:);
    xval.y_train = datay(:,itrn{1});
    [~,conf_matrix,v_error(i)] = Classifier.FisherLD_bin(xval,0);
    final_matrix=final_matrix+conf_matrix;
end

m = mean(v_error)
st = std(v_error)

Util.statistics(final_matrix);


%%
crossvalidation(kwb, 'Classifier.FisherLD_bin');