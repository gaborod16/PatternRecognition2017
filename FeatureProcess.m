classdef FeatureProcess
    %FEATUREPROCESS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
    end
    
    methods(Static)
        
        % Pearson correlation
        function [new_data, new_meta] = RemCorrelated(data,meta)
            R = corrcoef(data.X_train);
            new_IX = [];
            for i = 1:size(R,1)
                counter = 0;
                for j = i+1:size(R,1)
                    if(R(i,j) > 0.95)
                        %new_IX = [new_IX, j];
                        counter = 1;
                    end
                end
                if(counter == 0)
                    new_IX = [new_IX, i];
                end
            end

            %new_IX = unique(new_IX);
            new_data = data;
            new_data.X_train = data.X_train(:,new_IX);
            new_data.X_test = data.X_test(:,new_IX);
            new_meta = meta;
            new_meta.n_features = numel(new_IX);
            new_meta.features = meta.features(new_IX);
        end
        
        % Scaling
        function [scaled_data] = SimpleScale(data)
            scaled_data = data;
            for i=1:size(data.X_train, 2) %n features
                scaled_data.X_train(:,i) = -1 + 2.* (data.X_train(:,i)-min(data.X_train(:,i))) ./ (max(data.X_train(:,i)-min(data.X_train(:,i))));
                scaled_data.X_test(:,i) = -1 + 2.*(data.X_test(:,i)-min(data.X_test(:,i))) ./ (max(data.X_test(:,i)-min(data.X_test(:,i))));
            end
        end
        
        function [data_scaled] = StdScale(data)
            data_scaled = data;
            n_train_samples = numel(data.y_train);
            n_test_samples = numel(data.y_test);
            
            mu = mean(data.X_train', 2);
            s = std(data.X_train', [], 2);
            data_scaled.X_train = (data.X_train' - repmat(mu, 1 , n_train_samples))/repmat(s,1,n_train_samples)';
            
            mu = mean(data.X_test', 2);
            s = std(data.X_test', [], 2);
            data_scaled.X_test = (data.X_test' - repmat(mu, 1 , n_test_samples))/repmat(s,1,n_test_samples)';
        end
        
        function [walking_data, not_walking_data] = divide_and_conquer(result, data)
            idx_walking_train = result.train == 1;
            idx_not_walking_train = find(result.train ~= 1);
            idx_walking_test = result.test == 1;
            idx_not_walking_test = find(result.test ~= 1);
            
            % Walking
            walking_data = data;
            walking_data.X_train = data.X_train(idx_walking_train,:);
            walking_data.y_train = data.y_train(idx_walking_train);
            walking_data.X_test = data.X_test(idx_walking_test,:);
            walking_data.y_test = data.y_test(idx_walking_test);
            
            %Not Walking
            not_walking_data = data;
            not_walking_data.X_train = data.X_train(idx_not_walking_train,:);
            not_walking_data.y_train = data.y_train(idx_not_walking_train) - 3;
            not_walking_data.X_test = data.X_test(idx_not_walking_test,:);
            not_walking_data.y_test = data.y_test(idx_not_walking_test) - 3;
            
        end
        
        % Kruskal Wallis method.
        % Usage: FeatureProcess.KruskalWallis(data, meta, 5, 1);
        function [new_data] = KruskalWallis(data, meta, n_wanted_features, binary)
            
            if (binary)
                y_train = data.y_train_bin;
                y_test = data.y_test_bin;
            else
                y_train = data.y_train;
                y_test = data.y_test;
            end
            
            rank=cell(meta.n_features,2);

            for i=1:meta.n_features
                [p,atab,stats] = kruskalwallis(data.X_train(:,i),y_train,'off');
                rank{i,1}=meta.features{i};
                rank{i,2}=atab{2,5};
            end

            [~,I]=sort([rank{:,2}],2,'descend');
            stotal=[sprintf('K-W Feature ranking: \n')];
            for i=1:meta.n_features
                stotal= [stotal, sprintf('%s --> %.2f\n', rank{I(i),1}, rank{I(i),2})];
            end

            % stotal % Shows all the ranking

            new_data = struct();
            new_data.n_features = n_wanted_features;
            new_data.X_train = [];
            new_data.X_test = [];
            new_data.y_train = y_train;
            new_data.y_test = y_test;
            sfeat = [sprintf('Top %d features \t\t\t\t\t\tIndex\n',n_wanted_features)];
            sfeat = [sfeat, sprintf('----------------------------\t\t------\n')];
            for i=1:n_wanted_features
                new_data.X_train = [new_data.X_train, data.X_train(:,I(i))];
                new_data.X_test = [new_data.X_test, data.X_test(:,I(i))];
                sfeat = [sfeat, sprintf('%s\t\t\t%d\n', rank{I(i),1}, I(i))];
            end

            sfeat % Shows the top 'n_wanted_features'
        end
        
        % Principal component analysis method.
        function [new_data] = PCA(data, n_wanted_features, binary)
            in_data.X = data.X_train';
            in_test_data.X = data.X_test';
            if (binary) 
                in_data.y = data.y_train_bin;
                in_test_data.y = data.y_test_bin;
            else
                in_data.y = data.y_train;
                in_test_data.y = data.y_test;
            end

            model = pca(in_data.X, n_wanted_features);
            in_data
            out_data = linproj(in_data,model);
            out_test_data = linproj(in_test_data,model);

            new_data.X_train = out_data.X';
            new_data.X_test = out_test_data.X';
            new_data.y_train = in_data.y;
            new_data.y_test = in_test_data.y;
            new_data.n_features = n_wanted_features;

%             figure; 
%             ppatterns(out_data);
%             figure; 
%             ppatterns(out_test_data);
        end
        
        % Linear discriminant analysis method for the binary scenario
        function [new_data] = LDA(data, n_wanted_features, binary)
            in_data.X = data.X_train';
            in_test_data.X = data.X_test';
            if (binary) 
                in_data.y = data.y_train_bin;
                in_test_data.y = data.y_test_bin;
            else
                in_data.y = data.y_train;
                in_test_data.y = data.y_test;
            end

            model = lda(in_data, n_wanted_features);

            out_data = linproj(in_data,model);
            out_test_data = linproj(in_test_data,model);
            
            new_data.X_train = out_data.X';
            new_data.X_test = out_test_data.X';
            new_data.y_train = in_data.y;
            new_data.y_test = in_test_data.y;
            new_data.n_features = n_wanted_features;
            
%             figure; 
%             ppatterns(out_data);
%             figure; 
%             ppatterns(out_test_data);
        end
    end
    
end

