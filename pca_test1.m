% The original data is too large. Extract 1/10 of the data from the
% original data as the training data
clear;
load('mnist_uint8.mat');
X=train_x(1:10:end,:); %selected the training data set
Y=train_y(1:10:end,:); % index of the class of the training data
K=20;% dimension reduction from 748 to 24
test_x=test_x';  % Test data set
test_y=test_y';
M=155;
Test1=test_x(:,M);%select only M test data sets from the original test data
[zz,b]=max(test_y);  
N=b(M); %index of the class of the test set

%input:
%X:Ten classes of input data  
%K: the number of eigenvectors and eigenvalues
%Test1: Test data 
%N: index of the class of data set
%Output:
%ind:the index of the selected class
U=zeros(784,K*10); Xmean=zeros(784,10);
% Training procedure using PCA
for i=1:10
    X1=X(logical(Y(:,i)),:);   
    [u,~]=pca(double(X1));
    U(:,(i-1)*K+1:(i*K))=u(:,1:K); %eigenvectors
    Xmean(:,i)=mean(X1);
end
%Testing procedure to find the fit
    test_data=double(Test1);
    e=zeros(1,10);
    for j=1:10
        EV=U(:,((j-1)*K+1):(j*K));
        xm=Xmean(:,j);
        score=EV'*(test_data-xm);
        xfit=EV*score+xm;
        e(j)=norm(test_data-xfit); %distance between the test data and each class of trained data
    end
    [~,ind]=min(e);%Find the minimum distance
EV=U(:,((ind-1)*K+1):(ind*K));
xm=Xmean(:,ind);
score=EV'*(test_data-xm);
xfit=EV*score+xm;    
figure (1); imshow(reshape(test_data,28,28))
figure (2); imshow(reshape(xfit,28,28))
fprintf('ind=%2d, Ln=%2d\n',ind,N)

