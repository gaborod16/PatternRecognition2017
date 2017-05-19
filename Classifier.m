classdef Classifier
    %CLASSIFIER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
    end
    
    methods(Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Fisher linear discriminant for the binary scenario %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [test_result,conf_matrix, error] = FisherLD_bin(feat_data, show)
            trn = struct();
            trn.X = feat_data.X_train'; % load training data
            trn.y = feat_data.y_train;

            [trn.dim, trn.num_data] = size(trn.X);
            trn.name = 'FLD';
            model = fld(trn); % compute FLD 
            if(show)
                figure; ppatterns(trn); %pline(model); 
                plane3(model);
                model
            end

            % plot data and solution 
            tst.X = feat_data.X_test'; % load testing data 
            tst.y = feat_data.y_test;

            test_result = linclass(tst.X,model); % classify testing data 
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
            
            model = fitcecoc(trn.X, trn.y, 'Learners', t, 'ObservationsIn', 'columns', 'Coding', 'onevsall', 'OptimizeHyperparameters', 'auto');
            test_result = predict(model, tst.X, 'ObservationsIn', 'columns');
            cerror(test_result, tst.y)
           
            % plot data and solution 
            if(show)
                figure; ppatterns(trn); %pline(model); 
%                 plane3(model);
            end
            
            conf_matrix=Util.confusion_matrix(test_result, tst.y, 1);
        end
    end
end