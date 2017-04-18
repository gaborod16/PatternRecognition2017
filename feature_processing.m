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

% Top 3 features                         Index
% -----------------------------          --------
% fBodyAccJerk-entropy()-X               1 ?!?!
% fBodyBodyAccJerkMag-entropy()          235 ?!?!
% fBodyAccJerk-entropy()-Y               104 !??!

% selected_features_train=[data.X_train(:,1), data.X_train(:,235), data.X_train(:,104)]; 
% selected_features_test=[data.X_test(:,1), data.X_test(:,235), data.X_test(:,104)];

clear stotal sfeat i rank p atab stats I;

%% Principal component analysis method for both scenario

% all=[X_train;X_test];
% [COEFF, SCORE, LATENT] = pca(all);
% SCORE_TRAIN=SCORE(1:7352,:);
% SCORE_TEST=SCORE(7353:end,:);
%1,2,3 walking
%4,5,6 not walking

[COEFF_TRAIN, SCORE_TRAIN, LATENT_TRAIN] = pca(data.X_train);
[COEFF_TEST, SCORE_TEST, LATENT_TEST] = pca(data.X_test);

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