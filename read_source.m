%% This script reads saves all the information in MatLab variables for future use.

% X TRAIN READ
file = fopen('./UCI_HAR_Dataset/train/X_train.txt', 'r');

X_train = [];
row = 1;

while ~feof(file)
    line = str2num(fgetl(file));
    for column = 1:561
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
    for column = 1:561
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