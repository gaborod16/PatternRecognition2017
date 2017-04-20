%% Binary classification KW + FisherLD
load read_source.mat;
kwb = FeatureProcess.KruskalWallis(data,meta,3,1);
Classifier.FisherLD_bin(kwb);

%% Binary classification KW + MinDistEuc
load read_source.mat;
kwb = FeatureProcess.KruskalWallis(data,meta,3,1);
Classifier.MinDistEuc(kwb);

%% Binary classification KW + MinDistMah
load read_source.mat;
kwb = FeatureProcess.KruskalWallis(data,meta,3,1);
Classifier.MinDistMah(kwb);

%% Binary classification PCA + FisherLD
load read_source.mat;
kwb = FeatureProcess.PCA(data,3,1);
Classifier.FisherLD_bin(kwb);

%% Binary classification PCA + MinDistEuc
load read_source.mat;
kwb = FeatureProcess.PCA(data,3,1);
Classifier.MinDistEuc(kwb);

%% Binary classification PCA + MinDistMah
load read_source.mat;
kwb = FeatureProcess.PCA(data,3,1);
Classifier.MinDistMah(kwb);

%% Binary classification LDA + FisherLD
load read_source.mat;
kwb = FeatureProcess.LDA(data,3,1);
Classifier.FisherLD_bin(kwb);

%% Binary classification LDA + MinDistEuc
load read_source.mat;
kwb = FeatureProcess.LDA(data,3,1);
Classifier.MinDistEuc(kwb);

%% Binary classification LDA + MinDistMah
load read_source.mat;
kwb = FeatureProcess.LDA(data,3,1);
Classifier.MinDistMah(kwb);

%% Multiclass classification KW + MinDistEuc
load read_source.mat;
kwb = FeatureProcess.KruskalWallis(data,meta,3,0);
Classifier.MinDistEuc(kwb);

%% Multiclass classification KW + MinDistMah
load read_source.mat;
kwb = FeatureProcess.KruskalWallis(data,meta,3,0);
Classifier.MinDistMah(kwb);

%% Multiclass classification PCA + MinDistEuc
load read_source.mat;
kwb = FeatureProcess.PCA(data,3,0);
Classifier.MinDistEuc(kwb);

%% Multiclass classification PCA + MinDistMah
load read_source.mat;
kwb = FeatureProcess.PCA(data,3,0);
Classifier.MinDistMah(kwb);