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
pcab = FeatureProcess.PCA(data,3,1);
Classifier.FisherLD_bin(pcab);

%% Binary classification PCA + MinDistEuc
load read_source.mat;
pcab = FeatureProcess.PCA(data,3,1);
Classifier.MinDistEuc(pcab);

%% Binary classification PCA + MinDistMah
load read_source.mat;
pcab = FeatureProcess.PCA(data,3,1);
Classifier.MinDistMah(pcab);

%% Binary classification LDA + FisherLD
load read_source.mat;
ldab = FeatureProcess.LDA(data,3,1);
Classifier.FisherLD_bin(ldab);

%% Binary classification LDA + MinDistEuc
load read_source.mat;
ldab = FeatureProcess.LDA(data,3,1);
Classifier.MinDistEuc(ldab);

%% Binary classification LDA + MinDistMah
load read_source.mat;
<<<<<<< HEAD
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
=======
ldab = FeatureProcess.LDA(data,3,1);
Classifier.MinDistMah(ldab);

%% Multiclass classification Kruskal + MinDistEuc
load read_source.mat;
kwm = FeatureProcess.KruskalWallis(data,meta,3,0);
Classifier.MinDistEuc(kwm);

%% Multiclass classification Kruskal + MinDistMah
load read_source.mat;
kwm = FeatureProcess.KruskalWallis(data,meta,3,0);
Classifier.MinDistMah(kwm);

%% Multiclass classification PCA + MinDistEuc
load read_source.mat;
pcam = FeatureProcess.PCA(data,3,0);
Classifier.MinDistEuc(pcam);

%% Multiclass classification PCA + MinDistMah
load read_source.mat;
pcam = FeatureProcess.PCA(data,3,0);
Classifier.MinDistMah(pcam);

%% Multiclass classification LDA + MinDistEuc
load read_source.mat;
ldam = FeatureProcess.LDA(data,3,0);
Classifier.MinDistEuc(ldam);

%% Multiclass classification LDA + MinDistMah
load read_source.mat;
ldam = FeatureProcess.LDA(data,3,0);
Classifier.MinDistMah(ldam);
>>>>>>> origin/master
