%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Name: Vaishnavi Kotha
%Course: ENGG*4660 - Medical Image Processing 
%Section: Section 01 - 11:30AM - 1:20PM
%Due Date: Thursday April 9, 2020 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
% Terminate any running programs or background operations, and clear memory 
clc
close all
clear all

%% READ IN IMAGE FILE 
% Reads in image file and assigns to corresponding variable
brodatz = double(imread('brodatz.tif'));

%% ALL FUNCTION CALLS

% STEP 1: Divide classes into blocks
blocks1024 = blockMaker(brodatz,32,32);

% STEP 2: Determine feature vector of all 1024 blocks
[featuresClasses] = callGetFeatureVector(blocks1024);

% STEP 3: Randomly assign 1 of 3 classes to each block and store is vector
% array of length # of blocks (1024 in this case)
randClasses = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]; % initializes vector of all possible classes
randClassArr = zeros(1,1024); % initializes random class array of size 1 x 1024

for i = 1:1024
    pos = randi(length(randClasses)); % randomizes position of each class
    randClassArr(i) = randClasses(pos); % assigns class to block based on value at randomized position
end

% STEP 4: Take 32 x 32 features array containing 1 x 10 vectors and 
% transform into 1024 x 10 features list
[featList] = makeFeaturesList(featuresClasses);

% STEP 5: Determine mean feature vectors of all 3 classes
[classMeanVectors] = getClassMeanVect(featList,randClassArr,randClasses);

% STEP 6: Compute Euclidean distance
[randClassArrFinal] = kmeans(featList,randClassArr,randClasses,classMeanVectors);

% STEP 7: Make Silhouette
makeSilhouette(featList,randClassArrFinal)

%% SPLITTING IMAGE INTO BLOCKS 
% Split up image into blocks characterized by length and width
% specifications upon function call
function [block] = blockMaker(img,xBlocks,yBlocks)
    [row,col] = size(img);
    
    % # of pixels in each block in x and y directions
    xSize = row/xBlocks;
    ySize = col/yBlocks;
    
    % Initializes array of ones of same size as # of pixels in each block
    % in x and y directions
    arrayX = xSize.*ones(xBlocks,1);
    arrayY = ySize.*ones(yBlocks,1);
    
    block = mat2cell(img,arrayX,arrayY); % creates cell structure of blocks characterized by length and width specifications
end

%% GATHERING TEN FEATURES FOR EACH BLOCK
% Extracts 10 features for each block and stores in 1 x 10 feature vector
function [features] = getFeatureVector(block)
    % Mean and standard deviation 
    features(1) = mean(block(:));
    features(2) = std(block(:));
    
    % Frequency domain shift to get means of higher frequency blocks
    fftBlock = abs(fftshift(fft2(block)));
   
    % 0 degrees
    features(3) = mean2(fftBlock(5:8,13:16));
    % 90 degrees 
    features(4) = mean2(fftBlock(1:4,9:12));
    % 45 degrees
    features(5) = mean2(fftBlock(1:4,13:16));
    % 135 degrees
    features(6) = mean2(fftBlock(1:4,1:4));
    
    % Horizontal direction - Takes sum of all products of each element (Tau
    % = 1) and of each fourth element (Tau = 4) iterating horizontally
    [row,col] = size(block);
    T1_h = 0;
    T4_h = 0;
    
    % For Tau = 1
    for i = 1:row
        for j = 2:col
            T1_h = (block(i,j) * block(i,j - 1)) + T1_h;
        end
    end
    % For Tau = 4
    for i = 1:row
        for j = 5:4:col
            T4_h = (block(i,j) * block(i,j - 4)) + T4_h;
        end
    end
    
    % Square roots horizontal sums
    features(7) = sqrt(T1_h);
    features(9) = sqrt(T4_h);
    
    % Vertical direction - Takes sum of all products of each element (Tau
    % = 1) and of each fourth element (Tau = 4) iterating vertically
    T1_v = 0;
    T4_v = 0;
    
    % For Tau = 1   
    for i = 1:col
        for j = 2:row
            T1_v = (block(j,i) * block(j - 1,i)) + T1_v;
        end
    end
    
    % For Tau = 4
    for i = 1:col
        for j = 5:4:row
            T4_v = (block(j,i) * block(j - 4,i)) + T4_v;
        end
    end
    
    % Square roots vertical sums
    features(8) = sqrt(T1_v);
    features(10) = sqrt(T4_v);
end

% Calls getFeatureVector function the same number of times as blocks within
% split image
function [allClassFeatures] = callGetFeatureVector(blockClass)
    [row,col] = size(blockClass);
    for i = 1:row
        for j = 1:col
            allClassFeatures{i,j} = getFeatureVector(blockClass{i,j});
        end
    end
end

% Restructures 32 x 32 cell of 1 x 10 feature vectors into 1024 x 10 array
% of feature vectors for each block
function [featList] = makeFeaturesList(featuresClasses)
    len = numel(featuresClasses);
    featList = zeros(len,10); % initializes features list
    
    x = 1;
    y = 0;
    
    % Restructures feature vectors cell array via two separate iterations 
    % due to different array sizes - one for iteration through the features 
    % list for updating and one for iteration through the feature vectors 
    % cell array
    for i = 1:len
        y = y + 1;
        
        for j = 1:10
           featList(i,j) = featuresClasses{x,y}(j);
        end
        
        % If end of row in block is reached, restart y counter for
        % interating through next row and update x counter to next row
        if y == 32
            x = x + 1;
            y = 0;
        end
    end
end

% Computes mean feature vector of each class
function [classMeanVectors] = getClassMeanVect(featList,randClassArr,randClasses)
    % Initializes each mean feature vector
    meanVectSum1 = zeros(1,10);
    meanVectSum2 = zeros(1,10);
    meanVectSum3 = zeros(1,10);
    meanVectSum4 = zeros(1,10);
    meanVectSum5 = zeros(1,10);
    meanVectSum6 = zeros(1,10);
    meanVectSum7 = zeros(1,10);
    meanVectSum8 = zeros(1,10);
    meanVectSum9 = zeros(1,10);
    meanVectSum10 = zeros(1,10);
    meanVectSum11 = zeros(1,10);
    meanVectSum12 = zeros(1,10);
    meanVectSum13 = zeros(1,10);
    meanVectSum14 = zeros(1,10);
    meanVectSum15 = zeros(1,10);
    meanVectSum16 = zeros(1,10);
    
    % Initializes counter
    count1 = 0;
    count2 = 0;
    count3 = 0;
    count4 = 0;
    count5 = 0;
    count6 = 0;
    count7 = 0;
    count8 = 0;
    count9 = 0;
    count10 = 0;
    count11 = 0;
    count12 = 0;
    count13 = 0;
    count14 = 0;
    count15 = 0;
    count16 = 0;
    
    % Computes summed feature vector of all blocks that are classified to
    % each of 16 classes and the number of instances (blocks) that are part
    % of the sum
    for i = 1:numel(randClassArr)
        if randClassArr(i) == randClasses(1)
            meanVectSum1 = featList(i,:) + meanVectSum1;
            count1 = count1 + 1;
        elseif randClassArr(i) == randClasses(2)
            meanVectSum2 = featList(i,:) + meanVectSum2;
            count2 = count2 + 1;
        elseif randClassArr(i) == randClasses(3)
            meanVectSum3 = featList(i,:) + meanVectSum3;
            count3 = count3 + 1;
        elseif randClassArr(i) == randClasses(4)
            meanVectSum4 = featList(i,:) + meanVectSum4;
            count4 = count4 + 1;
        elseif randClassArr(i) == randClasses(5)
            meanVectSum5 = featList(i,:) + meanVectSum5;
            count5 = count5 + 1;
        elseif randClassArr(i) == randClasses(6)
            meanVectSum6 = featList(i,:) + meanVectSum6;
            count6 = count6 + 1;
        elseif randClassArr(i) == randClasses(7)
            meanVectSum7 = featList(i,:) + meanVectSum7;
            count7 = count7 + 1;
        elseif randClassArr(i) == randClasses(8)
            meanVectSum8 = featList(i,:) + meanVectSum8;
            count8 = count8 + 1;
        elseif randClassArr(i) == randClasses(9)
            meanVectSum9 = featList(i,:) + meanVectSum9;
            count9 = count9 + 1;
        elseif randClassArr(i) == randClasses(10)
            meanVectSum10 = featList(i,:) + meanVectSum10;
            count10 = count10 + 1;
        elseif randClassArr(i) == randClasses(11)
            meanVectSum11 = featList(i,:) + meanVectSum11;
            count11 = count11 + 1;
        elseif randClassArr(i) == randClasses(12)
            meanVectSum12 = featList(i,:) + meanVectSum12;
            count12 = count12 + 1;
        elseif randClassArr(i) == randClasses(13)
            meanVectSum13 = featList(i,:) + meanVectSum13;
            count13 = count13 + 1;
        elseif randClassArr(i) == randClasses(14)
            meanVectSum14 = featList(i,:) + meanVectSum14;
            count14 = count14 + 1;
        elseif randClassArr(i) == randClasses(15)
            meanVectSum15 = featList(i,:) + meanVectSum15;
            count15 = count15 + 1;
        else 
            meanVectSum16 = featList(i,:) + meanVectSum16;
            count16 = count16 + 1;
        end
    end
    
    % Takes summed feature vector and divides by block count for each class
    % to obtain mean feature vector of each class
    classMeanVectors(1,:) = meanVectSum1/count1;
    classMeanVectors(2,:) = meanVectSum2/count2;
    classMeanVectors(3,:) = meanVectSum3/count3;
    classMeanVectors(4,:) = meanVectSum4/count4;
    classMeanVectors(5,:) = meanVectSum5/count5;
    classMeanVectors(6,:) = meanVectSum6/count6;
    classMeanVectors(7,:) = meanVectSum7/count7;
    classMeanVectors(8,:) = meanVectSum8/count8;
    classMeanVectors(9,:) = meanVectSum9/count9;
    classMeanVectors(10,:) = meanVectSum10/count10;
    classMeanVectors(11,:) = meanVectSum11/count11;
    classMeanVectors(12,:) = meanVectSum12/count12;
    classMeanVectors(13,:) = meanVectSum13/count13;
    classMeanVectors(14,:) = meanVectSum14/count14;
    classMeanVectors(15,:) = meanVectSum15/count15;
    classMeanVectors(16,:) = meanVectSum16/count16;
end

% Generates an updated array consisting of reassigned random classes for 
% each block
function [randClassArr] = kmeans(featList,randClassArr,randClasses,classMeanVectors)
    randClassArrOld = zeros(1,1024);
    
    % Loops until old random class array and new reassigned random class
    % array are the same
    while ~isequal(randClassArrOld,randClassArr)
        randClassArrOld = randClassArr;
        
        for j = 1:numel(randClassArr)
            dsm = inf; % Initializes minimum distance threshold as infinity
            for x = 1:numel(randClasses)
                % 1 x 1               1 x 10                                     10 x 1
                eucDist = (featList(j,:) - classMeanVectors(x,:))*(featList(j,:) - classMeanVectors(x,:))'; % Computes Euclidean distance between each of the 1024 blocks and all mean feature vectors of each class
                
                % Updates minimum distance threshold if Euclidean distance
                % is smaller than current minimum distance threshold
                if eucDist < dsm
                    randClassArr(j) = randClasses(x);
                    dsm = eucDist;
                end
            end
        end
        
        % Gets updated mean feature vectors for each class depending on
        % reassigned random class array for each block
        [classMeanVectors] = getClassMeanVect(featList,randClassArr,randClasses);
    end
end

% Generates silhouette using "silhouette" function
function makeSilhouette(featList,randClassArrFinal)
    figure()
    silhouette(featList,randClassArrFinal);
end