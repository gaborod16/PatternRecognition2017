%% Cross validation test
%%
crossvalidation(ldab, 2, 'Classifier.FisherLD');
%%
crossvalidation(pcam, 6, 'Classifier.Bayesian');
%%
crossvalidation(pcam, 6, 'Classifier.FisherLD');
%%
crossvalidation(ldab, 2, 'Classifier.SupportVM');
%% 
crossvalidation(kwm , 6, 'Classifier.SupportVM');

%%
crossvalidation(kwm , 6, 'Classifier.Bayesian');
%%
crossvalidation(kwm , 6, 'Classifier.KNearestNeighboors');

%%
crossvalidation(pcam , 6, 'Classifier.FisherLD');

%% 
crossvalidation(pcam , 6, 'Classifier.SupportVM');

%%
crossvalidation(pcam , 6, 'Classifier.Bayesian');

%% 
crossvalidation(pcam , 6, 'Classifier.KNearestNeighboors');

%% correr 
crossvalidation(pcam , 6, 'Classifier.HybridClassifier');

%% 
% (data, n_features, classifier for divide and conquer)
crossvalidation_dc(data, 75, 'Classifier.Bayesian');