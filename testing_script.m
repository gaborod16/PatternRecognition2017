%% Binary classification Kruskal + FisherLD
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data,meta);
kwb = FeatureProcess.KruskalWallis(data,meta,3,1);
Classifier.FisherLD(kwb,1);

%% Binary classification Kruskal + MinDistEuc
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
kwb = FeatureProcess.KruskalWallis(data,meta,3,1);
Classifier.MinDistEuc(kwb,1);

%% Binary classification Kruskal + MinDistMah
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
kwb = FeatureProcess.KruskalWallis(data,meta,3,1);
Classifier.MinDistMah(kwb,1);

%% Binary classification PCA + FisherLD
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcab = FeatureProcess.PCA(data,3,1);
Classifier.FisherLD(pcab,1);

%% Binary classification PCA + MinDistEuc
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcab = FeatureProcess.PCA(data,3,1);
Classifier.MinDistEuc(pcab,1);

%% Binary classification PCA + MinDistMah
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcab = FeatureProcess.PCA(data,3,1);
Classifier.MinDistMah(pcab,1);

%% Binary classification LDA + FisherLD [Perfect classification]
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
ldab = FeatureProcess.LDA(data,3,1);
Classifier.FisherLD(ldab,1);

%% Binary classification LDA + MinDistEuc
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
ldab = FeatureProcess.LDA(data,3,1);
Classifier.MinDistEuc(ldab,1);

%% Binary classification LDA + MinDistMah
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
ldab = FeatureProcess.LDA(data,3,1);
Classifier.MinDistMah(ldab,1);

%% Binary classification LDA + SVM [Perfect classification]
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
ldab = FeatureProcess.LDA(data,3,1);
Classifier.SupportVM(ldab,1);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%% Multiclass classification Kruskal + FisherLD
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data,meta);
kwb = FeatureProcess.KruskalWallis(data,meta,3,0);
Classifier.FisherLD(kwb,1);

%% Multiclass classification Kruskal + MinDistEuc
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
kwm = FeatureProcess.KruskalWallis(data,meta,3,0);
Classifier.MinDistEuc(kwm,1);

%% Multiclass classification Kruskal + MinDistMah
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
kwm = FeatureProcess.KruskalWallis(data,meta,3,0);
Classifier.MinDistMah(kwm,1);

%% Multiclass classification Kruskal + SVM
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
kwm = FeatureProcess.KruskalWallis(data,meta,3,0);
Classifier.SupportVM(kwm,1);

%% Multiclass classification Kruskal + Bayes
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
kwm = FeatureProcess.KruskalWallis(data,meta,3,0);
Classifier.Bayesian(kwm,1);

%% Multiclass classification Kruskal + KNN
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
kwm = FeatureProcess.KruskalWallis(data,meta,3,0);
Classifier.KNearestNeighboors(kwm,1);

%% Multiclass classification PCA + FisherLD
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcam = FeatureProcess.PCA(data,3,0);
Classifier.FisherLD(pcam,1);

%% Multiclass classification PCA + MinDistEuc
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcam = FeatureProcess.PCA(data,3,0);
Classifier.MinDistEuc(pcam,1);

%% Multiclass classification PCA + MinDistMah
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcam = FeatureProcess.PCA(data,3,0);
Classifier.MinDistMah(pcam,1);

%% Multiclass classification PCA + SVM
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcam = FeatureProcess.PCA(data,3,0);
Classifier.SupportVM(pcam,1);

%% Multiclass classification PCA + Bayes
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcam = FeatureProcess.PCA(data,3,0);
Classifier.Bayesian(pcam,1);

%% Multiclass classification PCA + KNN
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcam = FeatureProcess.PCA(data,3,0);
Classifier.KNearestNeighboors(pcam,1);

%% Multiclass classification PCA + Hybrid classifier
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcam = FeatureProcess.PCA(data,20,0);
Classifier.HybridClassifier(pcam,1);

%%
%%% LDA Doesn't work correctly for the multiclass scenario!
%%%
%% Multiclass classification LDA + MinDistEuc
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
ldam = FeatureProcess.LDA(data,3,0);
Classifier.MinDistEuc(ldam,1);

%% Multiclass classification LDA + SVM
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
ldam = FeatureProcess.LDA(data,3,0);
Classifier.SupportVM(ldam,1);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%% Divide and Conquer Classifiers
%% Binary classification LDA + FisherLD [Perfect classification]
%% Multiclass classification PCA + Bayes
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);

%Binary division
ldab = FeatureProcess.LDA(data,3,1);
[test_result, conf_matrix, error] = Classifier.FisherLD(ldab,1);

%Updating dataset based on results
bin_result = struct();
bin_result.test = test_result;
bin_result.train = data.y_train_bin;
[walking_data, not_walking_data] = FeatureProcess.divide_and_conquer(bin_result, data);

%Multiclass simplified division (2 predictions, 3 classes)
pcam1 = FeatureProcess.PCA(walking_data,20,0);
test_result1 = Classifier.HybridClassifier(pcam1,1);

pcam2 = FeatureProcess.PCA(not_walking_data,20,0);
test_result2 = Classifier.HybridClassifier(pcam2,1) + 3;

total_test_data = [walking_data.y_test, not_walking_data.y_test + 3];
total_test_result = [test_result1, test_result2];

total_error = cerror(total_test_result, total_test_data);
conf_matrix=Util.confusion_matrix(total_test_result, total_test_data, 1);
