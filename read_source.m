%% This script reads saves all the information in MatLab variables for future use.

% Classes
file = fopen('./UCI_HAR_Dataset/activity_labels.txt', 'r');
classes = {};
row = 1;
while ~feof(file)
    line = strsplit(fgetl(file), ' ');
    classes(row,1) = line(2);
    row = row + 1;
end
fclose(file);
n_classes = size(classes,1);

% Features
file = fopen('./UCI_HAR_Dataset/features.txt', 'r');
features = {};
row = 1;
while ~feof(file)
    line = strsplit(fgetl(file), ' ');
    features(row,1) = line(2);
    row = row + 1;
end
fclose(file);
n_features = size(features,1);

% X TRAIN READ
file = fopen('./UCI_HAR_Dataset/train/X_train.txt', 'r');
X_train = [];
row = 1;
while ~feof(file)
    line = str2num(fgetl(file));
    for column = 1:n_features
        X_train(row, column) = line(column);
    end
    row = row + 1;
end
fclose(file);

% Y TRAIN READ
file = fopen('./UCI_HAR_Dataset/train/y_train.txt', 'r');
y_train = [];
row = 1;
while ~feof(file)
    y_train(row) = str2num(fgetl(file));
    row = row + 1;
end
fclose(file);

% X TEST READ
file = fopen('./UCI_HAR_Dataset/test/X_test.txt', 'r');
X_test = [];
row = 1;
while ~feof(file)
    line = str2num(fgetl(file));
    for column = 1:n_features
        X_test(row, column) = line(column);
    end
    row = row + 1;
end
fclose(file);

% Y TEST READ
file = fopen('./UCI_HAR_Dataset/test/y_test.txt', 'r');
y_test = [];
row = 1;
while ~feof(file)
    y_test(row) = str2num(fgetl(file));
    row = row + 1;
end
fclose(file);

clear column file line row ans;


%% 
% all=[X_train;X_test];
% [COEFF, SCORE, LATENT] = pca(all);
% SCORE_TRAIN=SCORE(1:7352,:);
% SCORE_TEST=SCORE(7353:end,:);
%1,2,3 walking
%4,5,6 
[COEFF_TRAIN, SCORE_TRAIN, LATENT_TRAIN] = pca(X_train);
[COEFF_TEST, SCORE_TEST, LATENT_TEST] = pca(X_test);

%distancia à média
m_walking=[];
m_not_walking=[];

for i=1:7352
    if y_train(i)==1 || y_train(i)==2 || y_train(i)== 3
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

%% test
%1 walking
%2 not walking
test_result=[];
matrix=[0 0; 0 0];

for i=1:2947
    dist_w=sqrt((w_mean1-SCORE_TEST(i,1))^2+(w_mean2-SCORE_TEST(i,2))^2+(w_mean3-SCORE_TEST(i,3))^2);
    
    dist_n_w=sqrt((nw_mean1-SCORE_TEST(i,1))^2+(nw_mean2-SCORE_TEST(i,2))^2+(nw_mean3-SCORE_TEST(i,3))^2);
    
      
    if dist_w<dist_n_w
        test_result=[test_result 1];
    else
        test_result=[test_result 2];
    end
   
end

 for k=1:2947
     if test_result(k)==1 && (y_test(k)==1 || y_test(k)==2 || y_test(k)==3)
         matrix(1,1)=matrix(1,1)+1;
     elseif test_result(k)==2 && (y_test(k)==4 || y_test(k)==5 || y_test(k)==6)
         matrix(2,2)=matrix(2,2)+1;
     elseif test_result(k)==1 && (y_test(k)==4 || y_test(k)==5 || y_test(k)==6)
         matrix(2,1)=matrix(2,1)+1;
     elseif test_result(k)==2 && (y_test(k)==1 || y_test(k)==2 || y_test(k)==3)
         matrix(1,2)=matrix(1,2)+1;
     end
 end

 %% kruskal wallis
 y_train2=[];
for i=1:7352
    if y_train(i)==1 || y_train(i)==2 || y_train(i)== 3
        y_train2=[y_train2 1];
    else
       y_train2=[y_train2 2]; 
    end
    
end

rank=cell(561,2);

for i=1:561
    [p,atab,stats] = kruskalwallis(X_train(:,i),y_train2,'off');
    rank{i,1}=features{i};
    rank{i,2}=atab{2,5};
    
end

[Y,I]=sort([rank{:,2}],2,'descend');
stotal=[sprintf('K-W Feature ranking: \n')]
for i=1:561
    stotal= [stotal, sprintf('%s --> %.2f\n', rank{I(i),1}, rank{I(i),2})];
end
    
stotal


% fBodyAccJerk-entropy()-X        coluna 1
% fBodyBodyAccJerkMag-entropy()          235
% fBodyAccJerk-entropy()-Y               104

selected_features_train=[X_train(:,1), X_train(:,235),X_train(:,104)]; 
selected_features_test=[X_test(:,1), X_test(:,235),X_test(:,104)]; 

m_walking=[];
m_not_walking=[];

for i=1:7352
    if y_train2(i)==1 
        m_walking=[m_walking; selected_features_train(i,1:3)];
    else
        m_not_walking=[m_not_walking; selected_features_train(i,1:3)];
    end
    
end

w_mean1=mean(m_walking(:,1));
w_mean2=mean(m_walking(:,2));
w_mean3=mean(m_walking(:,3));

nw_mean1=mean(m_not_walking(:,1));
nw_mean2=mean(m_not_walking(:,2));
nw_mean3=mean(m_not_walking(:,3));

test_result=[];
matrix=[0 0; 0 0];

for i=1:2947
    dist_w=sqrt((w_mean1-selected_features_test(i,1))^2+(w_mean2-selected_features_test(i,2))^2+(w_mean3-selected_features_test(i,3))^2);
    
    dist_n_w=sqrt((nw_mean1-selected_features_test(i,1))^2+(nw_mean2-selected_features_test(i,2))^2+(nw_mean3-selected_features_test(i,3))^2);
    
      
    if dist_w<dist_n_w
        test_result=[test_result 1];
    else
        test_result=[test_result 2];
    end
   
end

 for k=1:2947
     if test_result(k)==1 && (y_test(k)==1 || y_test(k)==2 || y_test(k)==3)
         matrix(1,1)=matrix(1,1)+1;
     elseif test_result(k)==2 && (y_test(k)==4 || y_test(k)==5 || y_test(k)==6)
         matrix(2,2)=matrix(2,2)+1;
     elseif test_result(k)==1 && (y_test(k)==4 || y_test(k)==5 || y_test(k)==6)
         matrix(2,1)=matrix(2,1)+1;
     elseif test_result(k)==2 && (y_test(k)==1 || y_test(k)==2 || y_test(k)==3)
         matrix(1,2)=matrix(1,2)+1;
     end
 end

