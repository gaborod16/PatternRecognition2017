function [ final_matrix ] = crossvalidation_dc( feat_data, n_features, classifier_name )
%CROSSVALIDATION_DC
    % Cross validation test
    
    %Binary division
    ldab = FeatureProcess.LDA(feat_data,3,1);
    [test_result, ~, ~] = Classifier.FisherLD(ldab,1);

    %Updating dataset based on results
    bin_result = struct();
    bin_result.test = test_result;
    bin_result.train = feat_data.y_train_bin;
    [walking_data, not_walking_data] = FeatureProcess.divide_and_conquer(bin_result, feat_data);

    %Classifier
    classifier = str2func(classifier_name);

    %Multiclass divide and conquer (2 predictions, 3 classes)
    pcam1 = FeatureProcess.PCA(walking_data,n_features,0);
%     test_result1 = classifier(pcam1,1);

    pcam2 = FeatureProcess.PCA(not_walking_data,n_features,0);
%     test_result2 = classifier(pcam2,1) + 3;

%     total_test_data = [walking_data.y_test, not_walking_data.y_test + 3];
%     total_test_result = [test_result1, test_result2];

%     error = cerror(total_test_result, total_test_data);
%     conf_matrix=Util.confusion_matrix(total_test_result, total_test_data, 1);

    final_matrix1 = crossvalidation(pcam1, 3, 'Classifier.Bayesian');
    final_matrix2 = crossvalidation(pcam2, 3, 'Classifier.Bayesian');
    final_matrix = final_matrix1 + final_matrix2;
    
    Util.statistics(final_matrix);

end

