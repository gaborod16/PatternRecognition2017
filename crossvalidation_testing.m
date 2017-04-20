%% Cross validation test


dataX = kwb.X_train;
datay = kwb.y_train;
n_folds = 10;
n_data = numel(datay);
v_error = zeros(n_folds,1);

xval = kwb;
[itrn,itst] = crossval(n_data,n_folds);

for i=1:n_folds
    xval.X_test = dataX(itst{1},:);
    xval.y_test = datay(:,itst{1});
    xval.X_train = dataX(itrn{1},:);
    xval.y_train = datay(:,itrn{1});
    [~,v_error(i)] = Classifier.FisherLD_bin(xval,0);
end


mean(v_error)