% The original data is too large. Extract 1/10 of the data from the
% original data as the training data
clear all;
load('mnist_uint8.mat');
%tic
X=train_x(1:10:end,:); %selected the training data set
Y=train_y(1:10:end,:); % index of the class of the training data
K=24;% dimension reduction from 748 to 24
test_x=test_x';  % Test data set
test_y=test_y';
M=200;
Test1=test_x(:,1:M);%select only M test data sets from the original test data
[zz,b]=max(test_y);  
Lte=b(1,1:M); %index of the class of the test set
PCA_handwriting(X,Y,K,Test1,Lte);
%toc