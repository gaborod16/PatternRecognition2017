%% Cross validation test
%%
crossvalidation(ldab, 2, 'Classifier.FisherLD');
%%
crossvalidation(pcam, 6, 'Classifier.Bayesian');
%%
crossvalidation(pcam, 6, 'Classifier.FisherLD');