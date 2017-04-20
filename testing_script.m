%% Binary classification KW + FisherLD
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data,meta);
kwb = FeatureProcess.KruskalWallis(data,meta,3,1);
Classifier.FisherLD_bin(kwb,1);

%% Binary classification KW + MinDistEuc
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
kwb = FeatureProcess.KruskalWallis(data,meta,3,1);
Classifier.MinDistEuc(kwb,1);

%% Binary classification KW + MinDistMah
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
kwb = FeatureProcess.KruskalWallis(data,meta,3,1);
Classifier.MinDistMah(kwb,1);

%% Binary classification PCA + FisherLD
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
pcab = FeatureProcess.PCA(data,3,1);
Classifier.FisherLD_bin(pcab,1);

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

%% Binary classification LDA + FisherLD
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
ldab = FeatureProcess.LDA(data,3,1);
Classifier.FisherLD_bin(ldab,1);

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

%% Multiclass classification LDA + MinDistEuc
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
ldam = FeatureProcess.LDA(data,3,0);
Classifier.MinDistEuc(ldam,1);

%% Multiclass classification LDA + MinDistMah
load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data, meta);
ldam = FeatureProcess.LDA(data,3,0);
Classifier.MinDistMah(ldam,1);

