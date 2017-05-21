classdef Classifier
    %CLASSIFIER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
    end
    
    methods(Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Fisher linear discriminant %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [test_result,conf_matrix, error] = FisherLD(feat_data, show)
            classes = unique(feat_data.y_train);
            n_classes = numel(classes);
            trn = struct();
            trn.X = feat_data.X_train'; % load training data
            trn.y = feat_data.y_train;
            
            tst.X = feat_data.X_test'; % load testing data 
            tst.y = feat_data.y_test;

            [trn.dim, trn.num_data] = size(trn.X);
            trn.name = 'FLD';
            

            if n_classes == 2 %Binary
                model = fld(trn);
                if(show)
                    figure; ppatterns(trn); %pline(model); 
                    plane3(model);
                end
            else
                trn_c = trn;
                % - That + 1 = this class = 1 and not this class = 2.
                trn_c.y = 1 + (trn.y ~= 1);
                model = fld(trn_c);
                
                if(show)
                    figure; ppatterns(trn_c); %pline(model); 
                    plane3(model);
                end

                for c = 2:n_classes
                    trn_c = trn;
                    trn_c.y = 1 + (trn.y ~= c);
                    model_c = fld(trn_c);
                    model.b = [model.b; model_c.b];
                    model.W = [model.W model_c.W];
                    if(show)
                        figure; ppatterns(trn_c); %pline(model); 
                        plane3(model_c);
                    end
                end
            end
            
            [test_result, dfce] = linclass(tst.X,model); % classify testing data 
            error = cerror(test_result,tst.y);
           
            conf_matrix=Util.confusion_matrix(test_result, tst.y, 1);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Minimmum distance classifier Euclidean Distance %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [test_result,conf_matrix, error] = MinDistEuc(feat_data, show)         
            
            n_features = feat_data.n_features;
            classes = unique(feat_data.y_train);
            n_classes = numel(classes);
            data_X = feat_data.X_train';
            
            % -- Train -- %
            m = zeros(n_features, n_classes);

            for i = 1:n_features
                for c = 1:n_classes
                    m(i,c) = mean(data_X(i, find(feat_data.y_train==classes(c))));
                end
            end

            b = zeros(n_classes,1);

            for c = 1:n_classes
                b(c) = -0.5 * m(:,c)'*m(:,c); %Euclidean bias
            end
            model = struct('W', m, 'b', b, 'fun', 'linclass');
            
            % -- Test -- %
            test_result = linclass(feat_data.X_test',model); % classify testing data 
            error = cerror(test_result, feat_data.y_test);
            error
            
            % -> TODO We need to plot the hyperplane
            if (show)
                out.X = feat_data.X_train';
                out.y = feat_data.y_train;
                figure()
                ppatterns(out); 
%                 pline(model.W(:,1), model.b(1));
%               size(model.W)
%               size(model.b)
%               plane3(model);
            end

            conf_matrix=Util.confusion_matrix(test_result, feat_data.y_test, 1);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Minimmum distance classifier Mahalanobis distance %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [test_result,conf_matrix, error] = MinDistMah(feat_data, show)         
            
            n_features = feat_data.n_features;
            classes = unique(feat_data.y_train);
            n_classes = numel(classes);
            data_X = feat_data.X_train';
            data_X_test = feat_data.X_test';
            
            % -- Train -- %
            cm = cov(data_X');
            m = zeros(n_features, n_classes);

            for i = 1:n_features
                for c = 1:n_classes
                    m(i,c) = mean(data_X(i, find(feat_data.y_train==classes(c))));
                end
            end

            b = zeros(n_classes,1);

            for c = 1:n_classes
                b(c) = -0.5 * m(:,c)' * cm' * m(:,c); %Mahalanobis bias
            end
            
            model = struct('W', cm'*m, 'b', b);
            
            % -- Test -- %
            test_result = linclass(data_X_test,model); % classify testing data 
            error = cerror(test_result, feat_data.y_test);
            error

            if (show)
                % -> TODO We need to plot the hyperplane
                out.X = feat_data.X_train';
                out.y = feat_data.y_train;
                ppatterns(out); 
    %             pline(model);
            end
            
            conf_matrix=Util.confusion_matrix(test_result, feat_data.y_test, 1);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  Support Vector Machine with parameters training  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [test_result,conf_matrix, error] = SupportVM(feat_data, show) 
            trn = struct();
            trn.X = feat_data.X_train'; % load training data
            trn.y = feat_data.y_train;
            
            tst.X = feat_data.X_test'; % load testing data 
            tst.y = feat_data.y_test;

            [trn.dim, trn.num_data] = size(trn.X);
            trn.name = 'SVM';
            
            t = templateSVM('Standardize',1, 'IterationLimit', 100);
            model = fitcecoc(trn.X, trn.y, 'Learners', t, 'ObservationsIn', 'columns', 'Coding', 'onevsall', 'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, 'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',50, 'ShowPlots', show));
            test_result = predict(model, tst.X, 'ObservationsIn', 'columns');
            cerror(test_result, tst.y)
           
            % plot data and solution 
            if(show)
                figure; ppatterns(trn); %pline(model); 
            end
            
            conf_matrix=Util.confusion_matrix(test_result, tst.y, 1);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  Naive Bayes classifier  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [test_result, conf_matrix, error] = Bayesian(feat_data, show)
            classes = unique(feat_data.y_train);
            n_classes = numel(classes);
            trn = struct();
            trn.X = feat_data.X_train'; % load training data
            trn.y = feat_data.y_train;
            
            tst.X = feat_data.X_test'; % load testing data 
            tst.y = feat_data.y_test;

            [trn.dim, trn.num_data] = size(trn.X);
            trn.name = 'Bayes';
            
            model = mlcgmm(trn);
            
            for c = 1:n_classes
                idx = find(trn.y(:) == c);
                trn_c = trn;
                trn_c.X = trn.X(:,idx);
                trn_c.y = trn.y(idx);
                trn_c.num_data = numel(idx);
                model.Pclass{c} = mlcgmm(trn_c);
            end

            [test_result, dfce] = bayescls(tst.X,model);
            error = cerror(test_result, tst.y)
           
            % plot data and solution 
            if(show)
                figure; ppatterns(trn); %pline(model); 
            end
            
            conf_matrix=Util.confusion_matrix(test_result, tst.y, 1);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        %  K-nearest Neighboors  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        function [test_result,conf_matrix, error] = KNearestNeighboors(feat_data, show) 
            trn = struct();
            trn.X = feat_data.X_train'; % load training data
            trn.y = feat_data.y_train;
            
            tst.X = feat_data.X_test'; % load testing data 
            tst.y = feat_data.y_test;

            [trn.dim, trn.num_data] = size(trn.X);
            trn.name = 'KNN';
            
            model = fitcknn(trn.X', trn.y,'Standardize',1, 'OptimizeHyperparameters',{'NumNeighbors'}, 'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',50, 'ShowPlots', show));
            test_result = predict(model, tst.X')';
            cerror(test_result, tst.y)
           
            % plot data and solution 
            if(show)
                figure; ppatterns(trn); %pline(model); 
            end
            
            conf_matrix=Util.confusion_matrix(test_result, tst.y, 1);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  Hybrid clssifier: Bayes + KNN + MinDistEuc + Fisher One VS All  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [test_result,conf_matrix, error] = HybridClassifier(feat_data, show) 
            trn = struct();
            trn.X = feat_data.X_train'; % load training data
            trn.y = feat_data.y_train;
            
            tst.X = feat_data.X_test'; % load testing data 
            tst.y = feat_data.y_test;

            [trn.dim, trn.num_data] = size(trn.X);
            trn.name = 'Hybrid';
            
            test_results = Classifier.Bayesian(feat_data,0);
            test_result2 = Classifier.KNearestNeighboors(feat_data,0);
            test_result3 = Classifier.MinDistEuc(feat_data,0);
            test_result4 = Classifier.FisherLD(feat_data,0);
            
            test_results = [test_results; test_result2; test_result3; test_result4];
            
            test_result = mode(test_results);
            
            % plot data and solution 
            if(show)
                figure; ppatterns(trn); %pline(model); 
            end
            
            conf_matrix=Util.confusion_matrix(test_result, tst.y, 1);
        end
    end
end