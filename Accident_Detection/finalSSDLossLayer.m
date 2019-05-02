classdef finalSSDLossLayer < nnet.layer.ClassificationLayer
    % Example custom classification layer with sum of squares error loss.
    
    methods
        function layer = finalSSDLossLayer(name)
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'SSD Loss Layer';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Final SSD loss between
            % the predictions Y and the training targets T.

            % loss = sum(sumSquares)/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer, Y, T) returns the derivatives of
            % the Final SSD loss with respect to the predictions Y.

            % dLdY = 
        end
    end
end