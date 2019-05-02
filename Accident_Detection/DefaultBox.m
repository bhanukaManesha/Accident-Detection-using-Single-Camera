classdef DefaultBox
    properties
        feature_maps
        min_sizes
        max_sizes
        image_size
        has_six_default_boxes
    end
        
    methods
        function obj = DefaultBox()
            obj.image_size = 300;
            
            % Size of the feature maps
            obj.feature_maps = [38 19 10 5 3 1];
            
            % Calculated beforehand
            obj.min_sizes = [30 60 111 162 213 264];
            obj.max_sizes = [60 111 162 213 264 315];
           
            obj.has_six_default_boxes = [0 1 1 1 0 0];
        end
        
        % Generate all default boxes
        function boxes = forward(obj)
            % Hardcoded for now
            num_of_boxes = 8732;
            boxes = zeros(num_of_boxes, 4);
            
            counter = 1;
            obj.feature_maps
            for i = 1:size(obj.feature_maps, 2)
                map_size = obj.feature_maps(i);
                for j = 1:map_size
                    for k = 1:map_size
                        cx = (j - 0.5) / map_size;
                        cy = (k - 0.5) / map_size;
                    
                        % Aspect ratio: 1
                        % Size: small
                        s_k = obj.min_sizes(i) / obj.image_size;
                        boxes(counter, 1) = cx;
                        boxes(counter, 2) = cy;
                        boxes(counter, 3) = s_k;
                        boxes(counter, 4) = s_k;
                        counter = counter + 1;
                        
                        % Aspect ratio: 1
                        % Size: large
                        s_k_prime = sqrt(s_k * (obj.max_sizes(i) / obj.image_size));
                        boxes(counter, 1) = cx;
                        boxes(counter, 2) = cy;
                        boxes(counter, 3) = s_k_prime;
                        boxes(counter, 4) = s_k_prime;
                        counter = counter + 1;
                        
                        % Aspect ratio: 1/2
                        boxes(counter, 1) = cx;
                        boxes(counter, 2) = cy;
                        boxes(counter, 3) = s_k * sqrt(2);
                        boxes(counter, 4) = s_k / sqrt(2);
                        counter = counter + 1;
                        
                        % Aspect ratio: 2
                        boxes(counter, 1) = cx;
                        boxes(counter, 2) = cy;
                        boxes(counter, 3) = s_k / sqrt(2);
                        boxes(counter, 4) = s_k * sqrt(2);
                        counter = counter + 1;
                        
                        if obj.has_six_default_boxes(i) == 1
                            % Aspect ratio: 1/3
                            boxes(counter, 1) = cx;
                            boxes(counter, 2) = cy;
                            boxes(counter, 3) = s_k * sqrt(3);
                            boxes(counter, 4) = s_k / sqrt(3);
                            counter = counter + 1;

                            % Aspect ratio: 3
                            boxes(counter, 1) = cx;
                            boxes(counter, 2) = cy;
                            boxes(counter, 3) = s_k / sqrt(3);
                            boxes(counter, 4) = s_k * sqrt(3);
                            counter = counter + 1;                            
                        end
                    end
                end
            end
        end
    end
end