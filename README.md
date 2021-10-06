# Disclaimer
For the randos that stumble upon this, this is just some note taking I did when learning MatLab

## Summary
Created a fully connected neural network that can learn a set of inputs and outputs using backpropagation

## Implementation
Program consists of 4 files:
- `NeuralNetwork_Create`
- `NeuralNetwork_FeedForward`
- `NeuralNetwork_ComputeDeltas`
- `NeuralNetwork_ApplyDeltas`
- `NeuralNetwork_Train`

### NeuralNetwork_Create
```matlab
function networkWeightsLayersBias = NeuralNetwork_Create(structure)
```
This function creates a cell that contains a networks weights, nodes and biases from a given `structure` formated like so:
- `{ weights, nodes/layers, bias }`
`weights`, `nodes/layers`, `bias` are all cell arrays containing the corresponding matrices to represent the weights / nodes / bias of a given layer in the network

- For example, to access the first layers weights: `network{1}{1}` => first `{1}` gets the weights and the second `{1}` gets the first layer

The `structure` parameter is the node layout of the fully connected neural network
- Passed as an array such as `[ 2, 3, 2, 1 ]` which represents a network of input 2 (first value), 2 hidden layers of 3 nodes and 2 nodes respectively (middle values) and 1 output (final value)

#### Implementation

```matlab
function networkWeightsLayersBias = NeuralNetwork_Create(structure)

% Declare weights, biases and node layers containers
weights = cell(1, length(structure) - 1);
biases = cell(1, length(structure) - 1);
layers = cell(1, length(structure));

% Add nodes, weights and biases to the containers
for i = 1:length(structure)
    % Add nodes
    nodes = zeros(1, structure(i));
    layers{i} = nodes;
    
    % The first set of nodes are input nodes => skip and dont add biases / weights for them
    if (i ~= 1)
        % Add biases
        bias = rand(1, structure(i));
        biases{i - 1} = bias;

        % Add weights
        weight = rand(structure(i - 1), structure(i));
        weights{i - 1} = weight;
    end
end

% set return value
networkWeightsLayersBias = {weights, layers, biases};
```


### NeuralNetwork_FeedForward
```matlab
function [output, network] = NeuralNetwork_FeedForward(network, input)
```
This function takes in a `network` and calculates the output of the network for a given `input`
- `input` is in the format of a `1 x n` matrix / vector

#### Implementation
```matlab
function [output, network] = NeuralNetwork_FeedForward(network, input)

% Set input
network{2}{1} = input;

for i=2:(length(network{2}))
    if (i == 2)
		% Don't tanh() input
        network{2}{i} = network{2}{i-1} * network{1}{i-1} + network{3}{i - 1}; 
    else
		% Tanh previous nodes (activation function)
        network{2}{i} = tanh(network{2}{i-1}) * network{1}{i-1} + network{3}{i - 1};
    end
end

% return output (needs to be put through activation function)
% the node values left in the network cell object is used for
% backpropagation (needs to be the value prior activation)
output = tanh(network{2}{length(network{2})});
```


### NeuralNetwork_ComputeDeltas
```matlab
function deltas = NeuralNetwork_ComputeDeltas(network, output, expectedOutput)
```
This function calculates the neural network deltas from backpropagation.
- The returned `deltas` is a cell object that contains 2 cell objects which represent the deltas for the weights and biases respectively
	- `{weightdeltas, biasdeltas}` => of which the format of `weightdeltas` and `biasdeltas` follow the format of weights and biases from the network
- `network` => The neural network to get the deltas of
- `output` => The output of the neural network formatted as a 1xn matrix / vector
- `expectedOutput` => The expected / wanted output of the neural network formatted as a 1xn matrix / vector

*For it to work properly it requries the input for the associated output to be fed through the network with `NeuralNetwork_FeedForward` beforehand such that the node values used for backprop are available*

#### Implementation
```matlab
function deltas = NeuralNetwork_ComputeDeltas(network, output, expectedOutput)

weightdeltas = network{1};
biasdeltas = network{3};

% calculate cost derivative with output => uses squared error
layerDeltas = 2 * (output - expectedOutput);

% propagate backwards
for l=(length(network{2})):-1:2
    derivedActivationOutput = 1 - tanh(network{2}{l}).^2;
    gammaValues = derivedActivationOutput .* layerDeltas;
    for j=1:(size(network{2}{l}, 2))
        for i=1:(size(network{2}{l-1}, 2))
            weightdeltas{l-1}(i, j) = gammaValues(1, j) * network{2}{l-1}(1, i);
        end
    end
    biasdeltas{l-1} = gammaValues;
    layerDeltas = gammaValues * transpose(network{1}{l-1});
end

deltas = {weightdeltas, biasdeltas};
```

### NeuralNetwork_ApplyDeltas
```matlab
function network = NeuralNetwork_ApplyDeltas(network, deltas, learningRate)
```
This function applies a set of deltas to a neural network
- returns the `network` with the new deltas applied
- `network` => The neural network to get the deltas of
- `deltas` => The deltas to apply to the network
- `learningRate` => the strength at which the deltas should be applied

#### Implementation
```matlab
function network = NeuralNetwork_ApplyDeltas(network, deltas, learningRate)

for i=1:length(deltas{1})
    network{1}{i} = network{1}{i} - deltas{1}{i} * learningRate;
    network{3}{i} = network{3}{i} - deltas{2}{i} * learningRate;
end
```

### NeuralNetwork_Train
```matlab
function [network, cost] = NeuralNetwork_Train(network, inputs, outputs, learningRate)
```
This function performs 1 training batch on the provided neural network
- returns the new `network` after training and the `cost` of the network for the given batch
- `network` => neural network to train
- `inputs` => batch inputs
- `outputs` => batch outputs
- `learningRate` => learning rate for the training process

#### Implementation
```matlab
function [network, cost] = NeuralNetwork_Train(network, inputs, outputs, learningRate)

deltas = cell(1, length(inputs));

cost = 0;

for i=1:length(inputs)
    [output, network] = NeuralNetwork_FeedForward(network, inputs{i});
    cost = cost + (output - outputs{i}).^2;
    deltas{i} = NeuralNetwork_ComputeDeltas(network, output, outputs{i});
end

% Sum and average deltas

sumDeltas = deltas{1};
for i=2:length(inputs)
    for j=1:length(sumDeltas{1})
        sumDeltas{1}{j} = sumDeltas{1}{j} + deltas{i}{1}{j};
        sumDeltas{2}{j} = sumDeltas{2}{j} + deltas{i}{2}{j};
    end
end

for j=1:length(sumDeltas{1})
    sumDeltas{1}{j} = sumDeltas{1}{j} / length(inputs);
    sumDeltas{2}{j} = sumDeltas{2}{j} / length(inputs);
end

network = NeuralNetwork_ApplyDeltas(network, sumDeltas, learningRate);
```

## Example
Example to train a neural network to perform xor operations:

```matlab
inputs =  { [0, 0], [0, 1], [1, 0], [1, 1] };
outputs = { [0]   , [1]   , [1]   , [0]    };

network = NeuralNetwork_Create([ 2, 3, 2, 1 ]);

for i=1:10000
	[network, cost] = NeuralNetwork_Train(network, inputs, outputs, 0.1);
end

for i=1:length(inputs)
	[output, network] = NeuralNetwork_FeedForward(network, inputs{i});
	
	% Display input and network output
	inputs{i}
	output
end
```

Outputs:

```matlab
ans =
     0     0
output =
   2.8567e-04
ans =
     0     1
output =
    0.9871
ans =
     1     0
output =
    0.9873
ans =
     1     1
output =
   3.3002e-04
```
Which is pretty close the output of an xor

## MatLab features learned and used
- Utilized loops in MatLab including `for` and `while`
- Utilized cell arrays in MatLab to store data
- Utilized scripting and functions to create a neural network
- Utilized different matrix operators such as `.` operator for element-wise operations
- Utilized matrix initialization functions such as `ones()`, `zeros()` and `rand()`
- Utilized functions to get dimension sizes (`size()`) and lengths (`length()`) of cells and matrices
- Utilized MatLab's variable UI ( #MatLab-Workspace ) to debug code by peeking at values and verifying them
- Setup environment paths ( #MathLab-EnvironmentPaths ) in MatLab to access functions in other folders
## Improvements
The structure of the neural network can be improved:
- Use classes ( #MatLab-Classes ) rather than storing them as cell objects
	- This way all the functions can be converted to class methods making it a lot easier to use (don't need to pass around a network cell object)
	- Having weights, biases and layers stored as *properties* rather than in a cell object would also make the code much more readable
