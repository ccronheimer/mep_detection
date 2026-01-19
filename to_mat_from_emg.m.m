clc; clear all; close all; fclose('all');
[baseName, folder] = uigetfile();
completeFilePath = fullfile(folder, baseName);

dllFolder = 'C:\Program Files (x86)\Ripple\Trellis\Tools\thirdparty\nsNEVLibrary\x64';
setenv('PATH', [dllFolder ';' getenv('PATH')]);


dataType = 'Hi-Res';
%dataChan = [257 258 259 260 261 262 263];
dataChan = [129 130 131 132 133];

plotStatus = true;

% Open the file and extract basic information
[ns_status, hFile] = ns_OpenFile(completeFilePath);

if ns_status ~= 0
    error('Error opening the file. ns_status: %d', ns_status);
end

for ii = 1:length(dataChan)
    dataChannel = dataChan(ii);
    
    % Find the correct entity for the desired channel
    entityID = [];
    for i = 1:length(hFile.Entity)
        if hFile.Entity(i).ElectrodeID == dataChannel && strcmp(hFile.FileInfo(hFile.Entity(i).FileType).Type, 'nf3')
            entityID = i;
            break;
        end
    end
    
    if isempty(entityID)
        warning('Channel %d not found in the file. Skipping.', dataChannel);
        continue;
    end
    
    % Extract channel info
    [ns_RESULT, entityInfo] = ns_GetEntityInfo(hFile, entityID);
    
    if ns_RESULT ~= 0
        warning('Error getting entity info for channel %d. ns_RESULT: %d', dataChannel, ns_RESULT);
        continue;
    end
    
    % Extract analog info
    [ns_RESULT, analogInfo] = ns_GetAnalogInfo(hFile, entityID);
    
    if ns_RESULT ~= 0
        warning('Error getting analog info for channel %d. ns_RESULT: %d', dataChannel, ns_RESULT);
        continue;
    end
    
    % Extract data
    [ns_RESULT, count, analogData] = ns_GetAnalogData(hFile, entityID, 1, entityInfo.ItemCount);
    
    if ns_RESULT ~= 0
        warning('Error getting analog data for channel %d. ns_RESULT: %d', dataChannel, ns_RESULT);
        continue;
    end
    
    analogInputDataTime_s = (0:count-1)' / analogInfo.SampleRate;
    
    if plotStatus
        figure();
        plot(analogInputDataTime_s, analogData);
        xlabel('Time (s)');
        ylabel([dataType ' (uV)']);
        title(['Channel ' num2str(dataChannel)]);
    end
    
    % Save channel data
    chandata = analogData;
    save(fullfile(folder, ['chan' num2str(dataChannel) '.mat']), 'chandata');
end

% Save timestamps
save(fullfile(folder, 'Timestamps.mat'), 'analogInputDataTime_s');

% Close the file
ns_status = ns_CloseFile(hFile);

if ns_status ~= 0
    warning('Error closing the file. ns_status: %d', ns_status);
end
