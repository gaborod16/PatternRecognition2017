%% Fisher linear discriminant for the binary scenario
%  Using kruskal wallis features.


trn.X = kw_bin_f.X_train'; % load training data
trn.y = data.y_train_bin;
% a=find(trn.y(:)==0);
% b=find(trn.y(:)==1);
% trn.y(a)=1;
% trn.y(b)=2;

[trn.dim, trn.num_data] = size(trn.X);
trn.name = 'FLD';
model = fld(trn); % compute FLD 
figure; ppatterns(trn); pline(model); 

% plot data and solution 
tst.X = kw_bin_f.X_test'; % load testing data 
tst.y = data.y_test_bin;
% a=find(tst.y(:)==0);
% b=find(tst.y(:)==1);
% tst.y(a)=1;
% tst.y(b)=2;
ypred = linclass(tst.X,model); % classify testing data 
cerror(ypred,tst.y)

clear ans tst trn;
