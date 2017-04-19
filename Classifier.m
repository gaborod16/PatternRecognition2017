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
        function [output] = FisherLD_bin(feat_data)
            trn = struct();
            trn.X = feat_data.X_train'; % load training data
            trn.y = feat_data.y_train_bin;

            [trn.dim, trn.num_data] = size(trn.X);
            trn.name = 'FLD';
            model = fld(trn); % compute FLD 
            figure; ppatterns(trn); pline(model); 

            % plot data and solution 
            tst.X = feat_data.X_test'; % load testing data 
            tst.y = feat_data.y_test_bin;

            ypred = linclass(tst.X,model); % classify testing data 
            error = cerror(ypred,tst.y);
            
            output = [model, ypred, error];
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Minimmum distance classifier Euclidean Distance %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [output] = MinDistEuc(feat_data)         
            
            n_features = feat_data.n_features;
            classes = unique(feat_data.y_train);
            n_classes = numel(classes);
            data_X = feat_data.X_train';
            data_X_test = feat_data.X_test';
            
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
            b
            model = struct('W', m, 'b', b);
            
            % -- Test -- %
            test_result = linclass(data_X_test,model); % classify testing data 
            error = cerror(test_result, feat_data.y_test);
            error

            % -> TODO We need to plot the hyperplane

            Util.confusion_matrix(test_result, feat_data.y_test, 0);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Minimmum distance classifier Mahalanobis distance %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [output] = MinDistMah(feat_data)         
            
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
            
            cm
            
            b

            % -> TODO We need to plot the hyperplane

            Util.confusion_matrix(test_result, feat_data.y_test, 0);
        end
    end
end