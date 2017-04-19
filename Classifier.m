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
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Minimmum distance classifier for the binary scenario %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [output] = MinDistEuc_bin(feat_data)         
            
            n_train_samples = size(feat_data.y_train,2);
            n_test_samples = size(feat_data.y_test,2);
            n_features = feat_data.n_features;
            
            % Measures per class (walking and not walking)
            m_walking=[];
            m_not_walking=[];

            for i=1:n_train_samples %Nº train samples
                if feat_data.y_train(i) == 1
                    m_walking=[m_walking; feat_data.X_train(i,:)];
                else
                    m_not_walking=[m_not_walking; feat_data.X_train(i,:)];
                end
            end
            
            w_means = zeros(n_features,1);
            nw_means = zeros(n_features,1);
            for i=1:n_features
                w_means(i) = mean(m_walking(:,i));
                nw_means(i) = mean(m_not_walking(:,i));
            end
            
            % -- Test -- %
            test_result = [];

            for i=1:n_test_samples
                dist_w = Util.Euclidean_dist(w_means, feat_data.X_test(i,:));
                dist_n_w = Util.Euclidean_dist(nw_means, feat_data.X_test(i,:));

                if dist_w < dist_n_w
                    test_result=[test_result 1];
                else
                    test_result=[test_result 2];
                end
            end
            
            error = cerror(test_result,feat_data.y_test)
            %output=[model, test_result, error];

            % -> TODO We need to plot the hyperplane

            Util.confusion_matrix(test_result, feat_data.y_test, 1);
        end
        
        % Minimmum distance classifier for the multiclass scenario
        function [output] = MinDistEuc_mult(feat_data)
            
            n_train_samples = size(feat_data.y_train,2);
            n_test_samples = size(feat_data.y_test,2);
            n_features = feat_data.n_features;
            
            %distancia à média
            m_1=[];
            m_2=[];
            m_3=[];
            m_4=[];
            m_5=[];
            m_6=[];

            for i=1:n_train_samples
                if feat_data.y_train(i)==1 
                    m_1=[m_1; feat_data.X_train(i,:)];
                elseif feat_data.y_train(i)==2 
                    m_2=[m_2;feat_data.X_train(i,:)];
                elseif feat_data.y_train(i)==3
                    m_3=[m_3; feat_data.X_train(i,:)];
                elseif feat_data.y_train(i)==4
                    m_4=[m_4; feat_data.X_train(i,:)];
                elseif feat_data.y_train(i)==5
                    m_5=[m_5; feat_data.X_train(i,:)];
                elseif feat_data.y_train(i)==6
                    m_6=[m_6; feat_data.X_train(i,:)];   
                end

            end

            c1_means = zeros(n_features,1);
            c2_means = zeros(n_features,1);
            c3_means = zeros(n_features,1);
            c4_means = zeros(n_features,1);
            c5_means = zeros(n_features,1);
            c6_means = zeros(n_features,1);
            for i=1:n_features
                c1_means(i) = mean(m_1(:,i));
                c2_means(i) = mean(m_2(:,i));
                c3_means(i) = mean(m_3(:,i));
                c4_means(i) = mean(m_4(:,i));
                c5_means(i) = mean(m_5(:,i));
                c6_means(i) = mean(m_6(:,i));
            end

            % -- Test -- %
            test_result6=[];
            matrix6=zeros(6);

            for i=1:n_test_samples
                
                dist_c1 = Util.Euclidean_dist(c1_means, feat_data.X_test(i,:));
                dist_c2 = Util.Euclidean_dist(c2_means, feat_data.X_test(i,:));
                dist_c3 = Util.Euclidean_dist(c3_means, feat_data.X_test(i,:));
                dist_c4 = Util.Euclidean_dist(c4_means, feat_data.X_test(i,:));
                dist_c5 = Util.Euclidean_dist(c5_means, feat_data.X_test(i,:));
                dist_c6 = Util.Euclidean_dist(c6_means, feat_data.X_test(i,:)); 

                [~,I]=min([dist_c1,dist_c2,dist_c3,dist_c4,dist_c5,dist_c6]);

                test_result6 = [test_result6 I];

            end
            
            error = cerror(test_result,feat_data.y_test)
            Util.confusion_matrix(test_result6, feat_data.y_test, 1);
        end
    end
    
end

