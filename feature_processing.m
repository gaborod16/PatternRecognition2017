%% Scaling

for i=1:561
    data.X_train(:,i) =-1 + 2.* (data.X_train(:,i)-min(data.X_train(:,i))) ./ (max(data.X_train(:,i)-min(data.X_train(:,i))));
    data.X_test(:,i) = -1 + 2.*(data.X_test(:,i)-min(data.X_test(:,i))) ./ (max(data.X_test(:,i)-min(data.X_test(:,i))));
end

%% Kruskal Wallis method for the binary scenario.

rank=cell(meta.n_features,2);

for i=1:meta.n_features
    [p,atab,stats] = kruskalwallis(data.X_train(:,i),data.y_train_bin,'off');
    rank{i,1}=meta.features{i};
    rank{i,2}=atab{2,5};
    
end

[~,I]=sort([rank{:,2}],2,'descend');
stotal=[sprintf('K-W Feature ranking: \n')];
for i=1:meta.n_features
    stotal= [stotal, sprintf('%s --> %.2f\n', rank{I(i),1}, rank{I(i),2})];
end

% stotal % Shows all the ranking

kw_bin_f = struct();
kw_bin_f.n_features = 3;
kw_bin_f.X_train = [];
kw_bin_f.X_test = [];
sfeat = [sprintf('Top %d features \t\t\t\t\t\tIndex\n',kw_bin_f.n_features)];
sfeat = [sfeat, sprintf('----------------------------\t\t------\n')];
for i=1:kw_bin_f.n_features
    kw_bin_f.X_train = [kw_bin_f.X_train, data.X_train(:,I(i))];
    kw_bin_f.X_test = [kw_bin_f.X_test, data.X_test(:,I(i))];
    sfeat = [sfeat, sprintf('%s\t\t\t%d\n', rank{I(i),1}, I(i))];
end

sfeat % Shows the top 'n_wanted_features'

clear stotal sfeat i rank p atab stats I;


%% Principal component analysis method for both scenarios

n_features = 3;
in_data.X = data.X_train';
in_test_data.X = data.X_test';
model = pca(in_data.X, n_features);

% Binary scenario
in_data.y = data.y_train_bin;
in_test_data.y = data.y_test_bin;
out_data = linproj(in_data,model);
out_test_data = linproj(in_test_data,model);

pca_bin_f.X_train = out_data.X';
pca_bin_f.X_test = out_test_data.X';
pca_bin_f.n_features = n_features;

figure; 
ppatterns(out_data);
figure;
ppatterns(out_test_data);

% Multiclass scenario
in_data.y = data.y_train;
in_test_data.y = data.y_test;
out_data = linproj(in_data,model);
out_test_data = linproj(in_test_data,model);

pca_f.X_train = out_data.X';
pca_f.X_test = out_test_data.X';
pca_f.n_features = n_features;

figure; 
ppatterns(out_data);
figure; 
ppatterns(out_test_data);

clear in_data in_test_data model n_features out_data out_test_data;

%% Linear discriminant analysis method for the binary scenario

in_data.X = data.X_train';
in_data.y = data.y_train_bin;
model = lda(in_data, 3);

out_data = linproj(in_data,model);
figure; 
ppatterns(out_data);

%% Linear discriminant analysis method for the multiclass scenario

in_data.X = data.X_train';
in_data.y = data.y_train;
model = lda(in_data, 3);

out_data = linproj(in_data,model);
figure; 
ppatterns(out_data);