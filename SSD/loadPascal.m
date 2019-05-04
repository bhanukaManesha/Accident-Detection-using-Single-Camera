function imdb = loadPascal(varargin)
  opts.edition = '07' ;
%   opts.dataDir = fullfile('data','voc07') 
%   opts.archiveDir = fullfile('data','archives') ;
  opts.dataDir = fullfile("/Users/bhanuka/Documents/GitHub.nosync/SSD/kitti_dataset/", 'VOC2012') ;
  opts.includeDetection = true ;
  opts.includeSegmentation = false ;
  opts.includeTest = false ;
  opts.includeDevkit = false ;
  opts = vl_argparse(opts, varargin) ;

  % Download data
  if ~exist(fullfile(opts.dataDir,'Annotations'), 'dir'), download(opts) ; end

  % Source images and classes
  imdb.paths.image = esc(fullfile(opts.dataDir, 'JPEGImages', '%s.png')) ;
  imdb.sets.id = uint8([1 2 3]) ;
  imdb.sets.name = {'train', 'val', 'test'} ;
  imdb.classes.id = uint8(1:9) ;
  imdb.classes.name = {'dontcare','car','tram','cyclist','van','truck','person','person_sitting','misc'} ;
  imdb.classes.images = cell(1,9) ;
  imdb.images.id = [] ;
  imdb.images.name = {} ;
  imdb.images.set = [] ;
  index = containers.Map() ;
  [imdb, index] = addImageSet(opts, imdb, index, 'train', 1) ;
  [imdb, index] = addImageSet(opts, imdb, index, 'val', 2) ;
  if opts.includeTest, [imdb, index] = addImageSet(opts, imdb, index, 'test', 3) ; end

  % Source segmentations
%   if opts.includeSegmentation
%     n = numel(imdb.images.id) ;
%     imdb.paths.objectSegmentation = esc(fullfile(opts.dataDir, ...
%                                             'SegmentationObject', '%s.png')) ;
%     imdb.paths.classSegmentation = esc(fullfile(opts.dataDir, ...
%                                              'SegmentationClass', '%s.png')) ;
%     imdb.images.segmentation = false(1, n) ;
%     [imdb, index] = addSegmentationSet(opts, imdb, index, 'train', 1) ;
%     [imdb, index] = addSegmentationSet(opts, imdb, index, 'val', 2) ;
%     if opts.includeTest 
%       [imdb, ~] = addSegmentationSet(opts, imdb, index, 'test', 3) ; 
%     end
%   end

  % Compress data types
  imdb.images.id = uint32(imdb.images.id) ;
  imdb.images.set = uint8(imdb.images.set) ;
  for i=1:8
    imdb.classes.images{i} = uint32(imdb.classes.images{i}) ;
  end

  % Source detections
  if opts.includeDetection
    imdb.aspects.id = uint8(1:5) ;
    imdb.aspects.name = {'front', 'rear', 'left', 'right', 'misc'} ;
    imdb = addDetections(opts, imdb) ;
  end
  
  % Check images on disk and get their size
  imdb = getImageSizes(imdb) ;
  imdb.annotations = loadAnnotations(imdb,opts);

% -------------------------------------------------------------------------
function download(opts)
% -------------------------------------------------------------------------

  baseUrl = 'http://host.robots.ox.ac.uk:8080' ;

  switch opts.edition
    case '07'
      endpoint = sprintf('%s/pascal/VOC/voc2007', baseUrl) ;
      url = sprintf('%s/VOCtrainval_06-Nov-2007.tar', endpoint) ;
      testUrl = sprintf('%s/VOCtest_06-Nov-2007.tar', endpoint) ;
      devkitUrl = sprintf('%s/VOCdevkit_08-Jun-2007.tar', endpoint) ;
    case '08'
      url='' ;
    case '09'
      url='' ;
    case '10'
      url=sprintf('%s/pascal_trainval/VOCtrainval_03-May-2010.tar', baseUrl) ;
    case '11'
      url=sprintf('%s/pascal_trainval/VOCtrainval_25-May-2011.tar', baseUrl) ;
    case '12'
      base12 = sprintf('%s/pascal/VOC', baseUrl) ; % different for 2012
      url = sprintf('%s/voc2012/VOCtrainval_11-May-2012.tar', base12) ;
      devkitUrl = sprintf('%s/voc2012/VOCdevkit_18-May-2011.tar', base12) ;
  end

  trainvalData = sprintf('VOC20%strainval.tar', opts.edition) ;
  testData = sprintf('VOC20%stest.tar', opts.edition) ;
  devkit = sprintf('VOC20%sdevkit.tar', opts.edition) ;

  if ~exist(opts.archiveDir, 'dir'), mkdir(opts.archiveDir) ; end
  archivePath=fullfile(opts.archiveDir, trainvalData) ;
  if ~exist(archivePath, 'file')
    fprintf('%s: downloading %s to %s\n', mfilename, url, archivePath) ;
    urlwrite(url, archivePath) ;
  end
  fprintf('%s: decompressing and rearranging %s\n', mfilename, archivePath) ;
  untar(archivePath, opts.dataDir) ;
  switch opts.edition
    case '11' % Decompresses in TrainVal/VOCdevkit, Test/VOCdevkit (sigh!)
      movefile(fullfile(opts.dataDir, 'TrainVal'), fullfile(opts.dataDir,'Test')) ;
    otherwise % Decomporesses directly in VOCdevkit
  end

  if opts.includeDevkit
    devkitDir = fileparts(opts.dataDir) ;
    archivePath = fullfile(opts.archiveDir, devkit) ;
    if ~exist(archivePath, 'file')
      fprintf('%s: downloading %s to %s\n', mfilename, devkitUrl, archivePath) ;
      urlwrite(devkitUrl, archivePath) ;
    end
    fprintf('%s: decompressing and rearranging %s\n', mfilename, archivePath) ;
    untar(archivePath, devkitDir) ;
    movefile(fullfile(devkitDir, 'VOCdevkit', '*'), devkitDir) ;
    rmdir(fullfile(devkitDir, 'VOCdevkit')) ;
  end

  if opts.includeTest
  archivePath = fullfile(opts.archiveDir, testData) ;
    if ~exist(archivePath, 'file')
      try 
        fprintf('attempting download %s to %s\n', testUrl, archivePath) ;
        urlwrite(testUrl, archivePath) ;
       catch
         error(['Cannot download the test data automatically. ', ...
           'Please download the file %s manually from the PASCAL test ', ...
            'server.'], archivePath) ;
       end
    end
    fprintf('%s: decompressing and rearranging %s\n', mfilename, archivePath) ;
    untar(archivePath, opts.dataDir) ;
  end

  switch opts.edition
    case '11'
      movefile(fullfile(opts.dataDir, 'Test', 'VOCdevkit', ...
                        sprintf('VOC20%s',opts.edition), '*'), opts.dataDir) ;
      rmdir(fullfile(opts.dataDir, 'Test'),'s') ;
    otherwise
      movefile(fullfile(opts.dataDir, 'VOCdevkit', ...
                        sprintf('VOC20%s',opts.edition), '*'), opts.dataDir) ;
      rmdir(fullfile(opts.dataDir, 'VOCdevkit'),'s') ;
  end

% -------------------------------------------------------------------------
function [imdb, index] = addImageSet(opts, imdb, index, setName, setCode)
% -------------------------------------------------------------------------
j = length(imdb.images.id) ;
annoPath = fullfile(opts.dataDir, 'ImageSets', 'Main', 'trainval.txt') ;
fprintf('%s: reading %s\n', mfilename, annoPath) ;
[names] = textread(annoPath,'%s'); 
for i=1:length(names)
    if ~index.isKey(names{i})
      j = j + 1 ;
      index(names{i}) = j ;
      imdb.images.id(j) = j ;
      imdb.images.set(j) = setCode ;
      imdb.images.name{j} = names{i} ;
      imdb.images.classification(j) = true ;
      imdb.images.path = "";
    else
      j = index(names{i}) ;
    end
%     if labels(i) > 0, imdb.classes.images{ci}(end+1) = j ; end
 end


% ------------------------------------------------------------------------------
function [imdb, index] = addSegmentationSet(opts, imdb, index, setName, setCode)
% ------------------------------------------------------------------------------
  segAnnoPath = fullfile(opts.dataDir, 'ImageSets', ...
                                          'Segmentation', [setName '.txt']) ;
  fprintf('%s: reading %s\n', mfilename, segAnnoPath) ;
  segNames = textread(segAnnoPath, '%s') ; %#ok
  j = numel(imdb.images.id) ;
  for i=1:length(segNames)
    if index.isKey(segNames{i})
      k = index(segNames{i}) ;
      imdb.images.segmentation(k) = true ;
      imdb.images.set(k) = setCode ;
    else
      j = j + 1 ;
      index(segNames{i}) = j ;
      imdb.images.id(j) = j ;
      imdb.images.set(j) = setCode ;
      imdb.images.name{j} = segNames{i} ;
      imdb.images.classification(j) = false ;
      imdb.images.segmentation(j) = true ;
    end
  end

% -------------------------------------------------------------------------
function imdb = getImageSizes(imdb)
% -------------------------------------------------------------------------
  for j=1:numel(imdb.images.id)
    info = imfinfo(sprintf(imdb.paths.image, imdb.images.name{j})) ;
    imdb.images.size(:,j) = uint16([info.Width ; info.Height]) ;
    msg = '%s: checked image %s [%d x %d]\n' ;
    fprintf(msg, mfilename, imdb.images.name{j}, info.Height, info.Width) ;
  end

% -------------------------------------------------------------------------
function imdb = addDetections(opts, imdb)
% -------------------------------------------------------------------------
  rois = {} ; k = 0 ; msg = '%s: getting detections for %d images\n' ;
  fprintf(msg, mfilename, numel(imdb.images.id)) ;
  for j=1:numel(imdb.images.id)
    fprintf('.') ; if mod(j,80)==0,fprintf('\n') ; end
    name = imdb.images.name{j} ;   
    annoPath = fullfile(opts.dataDir, 'Annotations', [name '.xml']) ;
    if ~exist(annoPath, 'file')
      if imdb.images.classification(j) && imdb.images.set(j) ~= 3
        msg = 'Could not find detection annotations for ''%s''. Skipping.' ;
        warning(msg, name) ;
      end
      continue ;
    end
    
    doc = xmlread(annoPath) ;
    x = parseXML(doc, doc.getDocumentElement()) ;
    for q = 1:numel(x.object)
      xmin = sscanf(x.object(q).bndbox.xmin,'%d') ;
      ymin = sscanf(x.object(q).bndbox.ymin,'%d') ;
      xmax = sscanf(x.object(q).bndbox.xmax,'%d') - 1 ;
      ymax = sscanf(x.object(q).bndbox.ymax,'%d') - 1 ;
      
      k = k + 1 ; roi.id = k ;
      roi.image = imdb.images.id(j) ;
      roi.class = find(strcmp(x.object(q).name, imdb.classes.name)) ;
      roi.box = [xmin;ymin;xmax;ymax] ;
      roi.difficult = logical(sscanf(x.object(q).difficult,'%d')) ;
      roi.truncated = logical(sscanf(x.object(q).truncated,'%d')) ;
      if isfield(x.object(q),'occluded')
        roi.occluded = logical(sscanf(x.object(q).occluded,'%d')) ;
      else
        roi.occluded = false ;
      end
      switch x.object(q).pose
        case 'frontal', roi.aspect = 1 ;
        case 'rear', roi.aspect = 2 ;
        case {'sidefaceleft', 'left'}, roi.aspect = 3 ;
        case {'sidefaceright', 'right'}, roi.aspect = 4 ;
        case {'','unspecified'}, roi.aspect = 5 ;
        otherwise, error('Unknown view ''%s''', x.object(q).pose) ;
      end
      rois{k} = roi ; %#ok
    end
  end
  fprintf('\n') ;

  rois = horzcat(rois{:}) ;
  imdb.objects = struct(...
      'id', uint32([rois.id]), ...
      'image', uint32([rois.image]), ...
      'class', uint8([rois.class]), ...
      'box', single([rois.box]), ...
      'difficult', [rois.difficult], ...
      'truncated', [rois.truncated], ...
      'occluded', [rois.occluded], ...
      'aspect', uint8([rois.aspect])) ;

% -------------------------------------------------------------------------
function value = parseXML(doc, x)
% -------------------------------------------------------------------------
  text = {''} ;
  opts = struct ;
  if x.hasChildNodes
    for c = 1:x.getChildNodes().getLength()
      y = x.getChildNodes().item(c-1) ;
      switch y.getNodeType()
        case doc.TEXT_NODE
          text{end+1} = lower(char(y.getData())) ; %#ok
        case doc.ELEMENT_NODE
          param = lower(char(y.getNodeName())) ;
          if strcmp(param, 'part'), continue ; end
          value = parseXML(doc, y) ;
          if ~isfield(opts, param)
            opts.(param) = value ;
          else
            opts.(param)(end+1) = value ;
          end
      end
    end
    if numel(fieldnames(opts)) > 0
      value = opts ;
    else
      value = strtrim(horzcat(text{:})) ;
    end
  end

% -------------------------------------------------------------------------
function str=esc(str)
% -------------------------------------------------------------------------
  str = strrep(str, '\', '\\') ;
% ------------------------------------------------
function annotations = loadAnnotations(imdb, opts)
% ------------------------------------------------
  annotations = cell(1, numel(imdb.images.name)) ;
  for ii = 1:numel(imdb.images.name)
    match = find(imdb.objects.image == ii) ;
%     if opts.excludeDifficult
%       keep = ~(imdb.objects.difficult(match)) ;
%     else
%       keep = 1:numel(match) ;
%     end
    keep = 1:numel(match) ;
    match = match(keep);
    boxes = imdb.objects.box(:,match) ;
    classes = imdb.objects.class(match) ;

    % normalize annotation
    imSize = repmat(imdb.images.size(:, ii)', [1 2]) ;
    gt.boxes = bsxfun(@rdivide, boxes', single(imSize)) ;
    gt.classes = classes + 1 ; % add offset for background
    assert(all(2 <= gt.classes) && all(gt.classes <= 21), ...
                 'pascal class labels do not lie in the expected range') ;
    annotations{ii} = gt ; 
    fprintf('Loading annotaton %d \n', ii) ;
  end