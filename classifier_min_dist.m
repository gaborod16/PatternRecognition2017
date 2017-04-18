%% Minimmum distance classifier for the binary scenario
%  Using Kruskal Wallis feature selection.

% Measures walking and not walking
m_walking=[];
m_not_walking=[];

for i=1:meta.n_train_samples
    if data.y_train_bin(i) == 1 
        m_walking=[m_walking; kw_bin_f.X_train(i,:)];
    else
        m_not_walking=[m_not_walking; kw_bin_f.X_train(i,:)];
    end
    
end

% Walking means by feature
w_mean1=mean(m_walking(:,1));
w_mean2=mean(m_walking(:,2));
w_mean3=mean(m_walking(:,3));

% Not walking means by feature
nw_mean1=mean(m_not_walking(:,1));
nw_mean2=mean(m_not_walking(:,2));
nw_mean3=mean(m_not_walking(:,3));


% -- Test -- %
test_result=[];

for i=1:meta.n_test_samples
    dist_w=sqrt((w_mean1-kw_bin_f.X_test(i,1))^2+(w_mean2-kw_bin_f.X_test(i,2))^2+(w_mean3-kw_bin_f.X_test(i,3))^2);
    dist_n_w=sqrt((nw_mean1-kw_bin_f.X_test(i,1))^2+(nw_mean2-kw_bin_f.X_test(i,1))^2+(nw_mean3-kw_bin_f.X_test(i,1))^2);

    if dist_w < dist_n_w
        test_result=[test_result 1];
    else
        test_result=[test_result 0];
    end
   
end

% -> TODO We need to plot the hyperplane

conf_matrix=[0 0; 0 0];
for k=1:meta.n_test_samples
    if test_result(k)==1 && data.y_test_bin(k)==1
        conf_matrix(1,1)=conf_matrix(1,1)+1;
    elseif test_result(k)==0 && data.y_test_bin(k)==0
        conf_matrix(2,2)=conf_matrix(2,2)+1;
    elseif test_result(k)==1 && data.y_test_bin(k)==0
        conf_matrix(2,1)=conf_matrix(2,1)+1;
    elseif test_result(k)==0 && data.y_test_bin(k)==1
        conf_matrix(1,2)=conf_matrix(1,2)+1;
    end
end

conf_matrix
sprintf('accuracy = %.4f',(conf_matrix(1,1) + conf_matrix(2,2)) / sum(sum(conf_matrix)))
sprintf('precision = %.4f', conf_matrix(1,1) / (conf_matrix(1,1) + conf_matrix(2,1)))
sprintf('recall = %.4f', conf_matrix(1,1) / (conf_matrix(1,1) + conf_matrix(1,2)))
 
 clear dist_n_w dist_w i k ans nw_mean1 nw_mean2 nw_mean3 w_mean1 w_mean2 w_mean3 m_not_walking m_walking;

 
%% Minimmum distance classifier for the binary scenario
%  Using PCA feature selection.

% Measures walking and not walking
m_walking=[];
m_not_walking=[];

for i=1:meta.n_train_samples
    if data.y_train_bin(i) == 1 
        m_walking=[m_walking; pca_bin_f.X_train(i,:)];
    else
        m_not_walking=[m_not_walking; pca_bin_f.X_train(i,:)];
    end
    
end

% Walking means by feature
w_mean1=mean(m_walking(:,1));
w_mean2=mean(m_walking(:,2));
w_mean3=mean(m_walking(:,3));

% Not walking means by feature
nw_mean1=mean(m_not_walking(:,1));
nw_mean2=mean(m_not_walking(:,2));
nw_mean3=mean(m_not_walking(:,3));


% -- Test -- %
test_result=[];

for i=1:meta.n_test_samples
    dist_w=sqrt((w_mean1-pca_bin_f.X_test(i,1))^2+(w_mean2-pca_bin_f.X_test(i,2))^2+(w_mean3-pca_bin_f.X_test(i,3))^2);
    dist_n_w=sqrt((nw_mean1-pca_bin_f.X_test(i,1))^2+(nw_mean2-pca_bin_f.X_test(i,1))^2+(nw_mean3-pca_bin_f.X_test(i,1))^2);

    if dist_w < dist_n_w
        test_result=[test_result 1];
    else
        test_result=[test_result 0];
    end
   
end

% -> TODO We need to plot the hyperplane

conf_matrix=[0 0; 0 0];
for k=1:meta.n_test_samples
    if test_result(k)==1 && data.y_test_bin(k)==1
        conf_matrix(1,1)=conf_matrix(1,1)+1;
    elseif test_result(k)==0 && data.y_test_bin(k)==0
        conf_matrix(2,2)=conf_matrix(2,2)+1;
    elseif test_result(k)==1 && data.y_test_bin(k)==0
        conf_matrix(2,1)=conf_matrix(2,1)+1;
    elseif test_result(k)==0 && data.y_test_bin(k)==1
        conf_matrix(1,2)=conf_matrix(1,2)+1;
    end
end

conf_matrix
sprintf('accuracy = %.4f',(conf_matrix(1,1) + conf_matrix(2,2)) / sum(sum(conf_matrix)))
sprintf('precision = %.4f', conf_matrix(1,1) / (conf_matrix(1,1) + conf_matrix(2,1)))
sprintf('recall = %.4f', conf_matrix(1,1) / (conf_matrix(1,1) + conf_matrix(1,2)))
 
 clear dist_n_w dist_w i k ans nw_mean1 nw_mean2 nw_mean3 w_mean1 w_mean2 w_mean3 m_not_walking m_walking;
 
%% Minimmum distance classifier for the binary scenario
%  Using PCA feature selection.

%distancia � m�dia
m_walking=[];
m_not_walking=[];

for i=1:meta.n_train_samples
    if y_train_bin(i) == 1
        m_walking=[m_walking; SCORE_TRAIN(i,1:3)];
    else
        m_not_walking=[m_not_walking; SCORE_TRAIN(i,1:3)];
    end
    
end

w_mean1=mean(m_walking(:,1));
w_mean2=mean(m_walking(:,2));
w_mean3=mean(m_walking(:,3));

nw_mean1=mean(m_not_walking(:,1));
nw_mean2=mean(m_not_walking(:,2));
nw_mean3=mean(m_not_walking(:,3));

% -- Test -- %
%1 walking
%0 not walking
test_result=[];
conf_matrix=[0 0; 0 0];

for i=1:n_test_samples
    dist_w=sqrt((w_mean1-SCORE_TEST(i,1))^2+(w_mean2-SCORE_TEST(i,2))^2+(w_mean3-SCORE_TEST(i,3))^2);
    
    dist_n_w=sqrt((nw_mean1-SCORE_TEST(i,1))^2+(nw_mean2-SCORE_TEST(i,2))^2+(nw_mean3-SCORE_TEST(i,3))^2);
    
      
    if dist_w<dist_n_w
        test_result=[test_result 1];
    else
        test_result=[test_result 2];
    end
   
end

 for k=1:meta.n_test_samples
     if test_result(k)==1 && data.y_test_bin(k)==1
         conf_matrix(1,1)=conf_matrix(1,1)+1;
     elseif test_result(k)==0 && data.y_test_bin(k)==0
         conf_matrix(2,2)=conf_matrix(2,2)+1;
     elseif test_result(k)==1 && data.y_test_bin(k)==0
         conf_matrix(2,1)=conf_matrix(2,1)+1;
     elseif test_result(k)==0 && data.y_test_bin(k)==1
         conf_matrix(1,2)=conf_matrix(1,2)+1;
     end
 end

%% Minimmum distance classifier for the multiclass scenario
%  Using PCA feature selection.
 
%distancia � m�dia
m_1=[];
m_2=[];
m_3=[];
m_4=[];
m_5=[];
m_6=[];

for i=1:meta.n_train_samples
    if data.y_train(i)==1 
        m_1=[m_1; SCORE_TRAIN(i,1:3)];
    elseif y_train(i)==2 
        m_2=[m_2; SCORE_TRAIN(i,1:3)];
    elseif y_train(i)==3
        m_3=[m_3; SCORE_TRAIN(i,1:3)];
    elseif y_train(i)==4
        m_4=[m_4; SCORE_TRAIN(i,1:3)];
    elseif y_train(i)==5
        m_5=[m_5; SCORE_TRAIN(i,1:3)];
    elseif y_train(i)==6
        m_6=[m_6; SCORE_TRAIN(i,1:3)];   
    end
 
end


c1_mean1=mean(m_1(:,1));
c1_mean2=mean(m_1(:,2));
c1_mean3=mean(m_1(:,3));
c2_mean1=mean(m_2(:,1));
c2_mean2=mean(m_2(:,2));
c2_mean3=mean(m_3(:,3));
c3_mean1=mean(m_3(:,1));
c3_mean2=mean(m_3(:,2));
c3_mean3=mean(m_3(:,3));
c4_mean1=mean(m_4(:,1));
c4_mean2=mean(m_4(:,2));
c4_mean3=mean(m_4(:,3));
c5_mean1=mean(m_5(:,1));
c5_mean2=mean(m_5(:,2));
c5_mean3=mean(m_5(:,3));
c6_mean1=mean(m_6(:,1));
c6_mean2=mean(m_6(:,2));
c6_mean3=mean(m_6(:,3));



% -- Test -- %
test_result6=[];
matrix6=zeros(6);

for i=1:meta.n_test_samples
    dist_c1=sqrt((c1_mean1-SCORE_TEST(i,1))^2+(c1_mean2-SCORE_TEST(i,2))^2+(c1_mean3-SCORE_TEST(i,3))^2);
    dist_c2=sqrt((c2_mean1-SCORE_TEST(i,1))^2+(c2_mean2-SCORE_TEST(i,2))^2+(c2_mean3-SCORE_TEST(i,3))^2);
    dist_c3=sqrt((c3_mean1-SCORE_TEST(i,1))^2+(c3_mean2-SCORE_TEST(i,2))^2+(c3_mean3-SCORE_TEST(i,3))^2);
    dist_c4=sqrt((c4_mean1-SCORE_TEST(i,1))^2+(c4_mean2-SCORE_TEST(i,2))^2+(c4_mean3-SCORE_TEST(i,3))^2);
    dist_c5=sqrt((c5_mean1-SCORE_TEST(i,1))^2+(c5_mean2-SCORE_TEST(i,2))^2+(c5_mean3-SCORE_TEST(i,3))^2);
    dist_c6=sqrt((c6_mean1-SCORE_TEST(i,1))^2+(c6_mean2-SCORE_TEST(i,2))^2+(c6_mean3-SCORE_TEST(i,3))^2);   
    
    [dist_min,I]=min([dist_c1,dist_c2,dist_c3,dist_c4,dist_c5,dist_c6]);
    
    if I==1
        test_result6=[test_result6 1];
    elseif I==2
        test_result6=[test_result6 2];
    elseif I==3
        test_result6=[test_result6 3];
    elseif I==4
        test_result6=[test_result6 4];
    elseif I==5
        test_result6=[test_result6 5];
    elseif I==6
        test_result6=[test_result6 6];
    end
   
end

 for k=1:meta.n_test_samples
     if test_result6(k)==1 && data.y_test(k)==1 
         matrix6(1,1)=matrix6(1,1)+1;
     elseif test_result6(k)==2 && data.y_test(k)==2
         matrix6(2,2)=matrix6(2,2)+1;
     elseif test_result6(k)==3 && data.y_test(k)==3
         matrix6(3,3)=matrix6(3,3)+1;
     elseif test_result6(k)==4 && data.y_test(k)==4
         matrix6(4,4)=matrix6(4,4)+1;
     elseif test_result6(k)==5 && data.y_test(k)==5
         matrix6(5,5)=matrix6(5,5)+1;
     elseif test_result6(k)==6 && data.y_test(k)==6
         matrix6(6,6)=matrix6(6,6)+1;
     end
 end