classdef Util
    %UTIL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
    end
    
    methods(Static)
        function distance = Euclidean_dist(v1, v2)
            v1 = v1(:)';
            v2 = v2(:)';
            distance = sqrt((v1-v2)*(v1-v2)');
        end
        
        function result = statistics(conf_matrix)
       

              
                sprintf('accuracy = %.4f',sum(diag(conf_matrix)) / sum(sum(conf_matrix)))
                if (size(conf_matrix,1) == 2)
                    sprintf('precision = %.4f', conf_matrix(1,1) / (conf_matrix(1,1) + conf_matrix(2,1)))
                    sprintf('sensitivity/recall = %.4f', conf_matrix(1,1) / (conf_matrix(1,1) + conf_matrix(1,2)))
                    sprintf('specificity = %.4f', conf_matrix(2,2) / (conf_matrix(2,2) + conf_matrix(2,1)))
                end
                
                conf_matrix
            
            
        end
        
        function conf_matrix = confusion_matrix(y_pred, y_test, show_out)
            conf_matrix = confusionmat(y_test, y_pred);
            
            if(show_out)  
                sprintf('accuracy = %.4f',sum(diag(conf_matrix)) / sum(sum(conf_matrix)))
                if (size(conf_matrix,1) == 2)
                    sprintf('precision = %.4f', conf_matrix(1,1) / (conf_matrix(1,1) + conf_matrix(2,1)))
                    sprintf('sensitivity/recall = %.4f', conf_matrix(1,1) / (conf_matrix(1,1) + conf_matrix(1,2)))
                    sprintf('specificity = %.4f', conf_matrix(2,2) / (conf_matrix(2,2) + conf_matrix(2,1)))
                end
                
                conf_matrix
            end
            
        end
    end
    
end

