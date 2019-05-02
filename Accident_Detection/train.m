% 
% 
% 
% % Function to train the SSD
% function success = train(ssd_graph)
    % Create a image datastore
    
image_path = strcat(pwd,'/test_ssd/images/training/');
% label_path = strcat(pwd,'/ssd_training/labels/');
imds = imageDatastore(image_path,'LabelSource','filenames');

% numTrainingFiles = imds.;
% [imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');





% 
% 
% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.01, ...
%     'MaxEpochs',4, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsValidation, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
% 
% net = trainNetwork(imdsTrain,ssd_graph,options);


% 
% end