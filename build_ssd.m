function ssd = build_ssd()
    num_classes = 21;

    net = vgg16;

    pool5 = maxPooling2dLayer(3, 'Stride', 1, 'Padding', [1 1 1 1], 'Name', 'pool5');
    conv6 = convolution2dLayer(3, 1024, 'Padding', [6 6 6 6], 'DilationFactor', 6, 'Name', 'conv6');
    conv7 = convolution2dLayer(1, 1024, 'Name', 'conv7');

    % Extract vgg up until relu5_3 (conv5_3 ?)
    vgg_base = [
        net.Layers(1:31)
        pool5
        conv6
        reluLayer('Name', 'relu6')
        conv7
        reluLayer('Name', 'relu7')
    ];

    ssd = [
        vgg_base
        convolution2dLayer(1, 256, 'Name', 'conv8_1')
        reluLayer('Name', 'relu8_1')
        convolution2dLayer(3, 512, 'Stride', 2, 'Padding', [1 1 1 1], 'Name', 'conv8_2')
        reluLayer('Name', 'relu8_2')
        convolution2dLayer(1, 128, 'Name', 'conv9_1')
        reluLayer('Name', 'relu9_1')
        convolution2dLayer(3, 256, 'Stride', 2, 'Padding', [1 1 1 1], 'Name', 'conv9_2')
        reluLayer('Name', 'relu9_2')
        convolution2dLayer(1, 128, 'Name', 'conv10_1')
        reluLayer('Name', 'relu10_1')
        convolution2dLayer(3, 256, 'Name', 'conv10_2')
        reluLayer('Name', 'relu10_2')
        convolution2dLayer(1, 128, 'Name', 'conv11_1')
        reluLayer('Name', 'relu11_1')
        convolution2dLayer(3, 256, 'Name', 'conv11_2')
        reluLayer('Name', 'relu11_2')
    ];

    ssd_graph = layerGraph(ssd);

    % 4 default boxes
    loc1 = convolution2dLayer(3, 4 * 4, 'Padding', [1 1 1 1], 'Name', 'loc1');
    conf1 = convolution2dLayer(3, 4 * num_classes, 'Padding', [1 1 1 1], 'Name', 'conf1');
    ssd_graph = addLayers(ssd_graph, loc1);
    ssd_graph = addLayers(ssd_graph, conf1);
    ssd_graph = connectLayers(ssd_graph, 'conv4_3', 'loc1');
    ssd_graph = connectLayers(ssd_graph, 'conv4_3', 'conf1');

    % 6 default boxes
    loc2 = convolution2dLayer(3, 6 * 4, 'Padding', [1 1 1 1], 'Name', 'loc2');
    conf2 = convolution2dLayer(3, 6 * num_classes, 'Padding', [1 1 1 1], 'Name', 'conf2');
    ssd_graph = addLayers(ssd_graph, loc2);
    ssd_graph = addLayers(ssd_graph, conf2);
    ssd_graph = connectLayers(ssd_graph, 'conv7', 'loc2');
    ssd_graph = connectLayers(ssd_graph, 'conv7', 'conf2');

    % 6 default boxes
    loc3 = convolution2dLayer(3, 6 * 4, 'Padding', [1 1 1 1], 'Name', 'loc3');
    conf3 = convolution2dLayer(3, 6 * num_classes, 'Padding', [1 1 1 1], 'Name', 'conf3');
    ssd_graph = addLayers(ssd_graph, loc3);
    ssd_graph = addLayers(ssd_graph, conf3);
    ssd_graph = connectLayers(ssd_graph, 'relu8_2', 'loc3');
    ssd_graph = connectLayers(ssd_graph, 'relu8_2', 'conf3');

    % 6 default boxes
    loc4 = convolution2dLayer(3, 6 * 4, 'Padding', [1 1 1 1], 'Name', 'loc4');
    conf4 = convolution2dLayer(3, 6 * num_classes, 'Padding', [1 1 1 1], 'Name', 'conf4');
    ssd_graph = addLayers(ssd_graph, loc4);
    ssd_graph = addLayers(ssd_graph, conf4);
    ssd_graph = connectLayers(ssd_graph, 'relu9_2', 'loc4');
    ssd_graph = connectLayers(ssd_graph, 'relu9_2', 'conf4');

    % 4 default boxes
    loc5 = convolution2dLayer(3, 4 * 4, 'Padding', [1 1 1 1], 'Name', 'loc5');
    conf5 = convolution2dLayer(3, 4 * num_classes, 'Padding', [1 1 1 1], 'Name', 'conf5');
    ssd_graph = addLayers(ssd_graph, loc5);
    ssd_graph = addLayers(ssd_graph, conf5);
    ssd_graph = connectLayers(ssd_graph, 'relu10_2', 'loc5');
    ssd_graph = connectLayers(ssd_graph, 'relu10_2', 'conf5');

    % 4 default boxes
    loc6 = convolution2dLayer(3, 4 * 4, 'Padding', [1 1 1 1], 'Name', 'loc6');
    conf6 = convolution2dLayer(3, 4 * num_classes, 'Padding', [1 1 1 1], 'Name', 'conf6');
    ssd_graph = addLayers(ssd_graph, loc6);
    ssd_graph = addLayers(ssd_graph, conf6);
    ssd_graph = connectLayers(ssd_graph, 'relu11_2', 'loc6');
    ssd_graph = connectLayers(ssd_graph, 'relu11_2', 'conf6');

    % concatanate the loc layers
    concat_loc = depthConcatenationLayer(6,'Name','concat_loc');
    ssd_graph = addLayers(ssd_graph, concat_loc);
    ssd_graph = connectLayers(ssd_graph, 'loc1', 'concat_loc/in1');
    ssd_graph = connectLayers(ssd_graph, 'loc2', 'concat_loc/in2');
    ssd_graph = connectLayers(ssd_graph, 'loc3', 'concat_loc/in3');
    ssd_graph = connectLayers(ssd_graph, 'loc4', 'concat_loc/in4');
    ssd_graph = connectLayers(ssd_graph, 'loc5', 'concat_loc/in5');
    ssd_graph = connectLayers(ssd_graph, 'loc6', 'concat_loc/in6');
    
    % concatanate the conf layers
    concat_conf = depthConcatenationLayer(6,'Name','concat_conf');
    ssd_graph = addLayers(ssd_graph, concat_conf);
    ssd_graph = connectLayers(ssd_graph, 'conf1', 'concat_conf/in1');
    ssd_graph = connectLayers(ssd_graph, 'conf2', 'concat_conf/in2');
    ssd_graph = connectLayers(ssd_graph, 'conf3', 'concat_conf/in3');
    ssd_graph = connectLayers(ssd_graph, 'conf4', 'concat_conf/in4');
    ssd_graph = connectLayers(ssd_graph, 'conf5', 'concat_conf/in5');
    ssd_graph = connectLayers(ssd_graph, 'conf6', 'concat_conf/in6');
    
    % concatanate all
    concat_all = depthConcatenationLayer(2,'Name','concat_all');
    ssd_graph = addLayers(ssd_graph, concat_all);
    ssd_graph = connectLayers(ssd_graph, 'concat_loc', 'concat_all/in1');
    ssd_graph = connectLayers(ssd_graph, 'concat_conf', 'concat_all/in2');
    
    plot(ssd_graph)
    
    %return the ssd graph
    ssd = ssd_graph;
end


