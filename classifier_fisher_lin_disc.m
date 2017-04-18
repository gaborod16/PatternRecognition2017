%% Fisher linear discriminant for the binary scenario
%  Using kruskal wallis features.

trn.X = kw_bin.X_train'; % load training data
trn.y = data.y_train_bin;
[trn.dim, trn.num_data] = size(trn.X);
trn.name = 'FLD';
model = fld(trn); % compute FLD 
figure; ppatterns(trn); pline(model); 

% plot data and solution 
tst.X = kw_bin.X_test'; % load testing data 
tst.y = data.y_test_bin;
ypred = linclass(tst.X,model); % classify testing data 
cerror(ypred,tst.y)

clear ans tst trn;
