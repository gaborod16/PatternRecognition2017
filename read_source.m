%% This script reads saves all the information in MatLab variables for future use.

data = struct();
meta = struct();

% Classes
file = fopen('./UCI_HAR_Dataset/activity_labels.txt', 'r');
meta.classes = {};
row = 1;
while ~feof(file)
    line = strsplit(fgetl(file), ' ');
    meta.classes(row,1) = line(2);
    row = row + 1;
end
fclose(file);
meta.n_classes = size(meta.classes,1);

% Features
file = fopen('./UCI_HAR_Dataset/features.txt', 'r');
meta.features = {};
row = 1;
while ~feof(file)
    line = strsplit(fgetl(file), ' ');
    meta.features(row,1) = line(2);
    row = row + 1;
end
fclose(file);
meta.n_features = size(meta.features,1);

% X TRAIN READ
file = fopen('./UCI_HAR_Dataset/train/X_train.txt', 'r');
data.X_train = [];
row = 1;
while ~feof(file)
    line = str2num(fgetl(file));
    for column = 1:meta.n_features
        data.X_train(row, column) = line(column);
    end
    row = row + 1;
end
fclose(file);

% Y TRAIN READ
file = fopen('./UCI_HAR_Dataset/train/y_train.txt', 'r');
data.y_train = [];
row = 1;
while ~feof(file)
    data.y_train(row) = str2num(fgetl(file));
    row = row + 1;
end
fclose(file);

meta.n_train_samples = size(data.y_train, 2);
% y_test > 3 => not walking => walking = 0 and not walking = 1
% That + 1 = walking = 1 and not walking = 2. 
data.y_train_bin = 1 + (data.y_train > 3);

% X TEST READ
file = fopen('./UCI_HAR_Dataset/test/X_test.txt', 'r');
data.X_test = [];
row = 1;
while ~feof(file)
    line = str2num(fgetl(file));
    for column = 1:meta.n_features
        data.X_test(row, column) = line(column);
    end
    row = row + 1;
end
fclose(file);

% Y TEST READ
file = fopen('./UCI_HAR_Dataset/test/y_test.txt', 'r');
data.y_test = [];
row = 1;
while ~feof(file)
    data.y_test(row) = str2num(fgetl(file));
    row = row + 1;
end
fclose(file);

meta.n_test_samples = size(data.y_test, 2);
% ((y_test > 3) => not walking). => walking = 0 and not walking = 1
% That + 1 = walking = 1 and not walking = 2. 
data.y_test_bin = 1 + (data.y_test > 3);

clear column file line row ans;
