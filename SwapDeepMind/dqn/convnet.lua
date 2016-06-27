--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"
require 'torch'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'pl'
require 'trepl'
require 'adaMax_binary_clip_shift'
require 'nn'
require 'SqrHingeEmbeddingCriterion'

require 'nn'
require './BinaryLinear.lua'

require './BinarizedNeurons'
if opt.type=='cuda' then
  require 'cunn'
  require 'cudnn'
end

local BatchNormalization;
if opt.SBN == true then
  require './BatchNormalizationShiftPow2'
  BatchNormalization = BatchNormalizationShiftPow2
else
  BatchNormalization = nn.BatchNormalization
end

function create_network(args)

    local net = nn.Sequential()
    local numHid =2048
    net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
 ---   local convLayer = nn.SpatialConvolution

 ---   net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
 ---                       args.filter_size[1], args.filter_size[1],
 ---                       args.filter_stride[1], args.filter_stride[1],1))
 ---   net:add(args.nl())

     net:add(BinaryLinear(784,numHid))
net:add(BatchNormalization(numHid, opt.runningVal))
net:add(nn.HardTanh())
net:add(BinarizedNeurons(opt.stcNeurons))
net:add(BinaryLinear(numHid,numHid,opt.stcWeights))
net:add(BatchNormalization(numHid, opt.runningVal))
net:add(nn.HardTanh())
net:add(BinarizedNeurons(opt.stcNeurons))
net:add(BinaryLinear(numHid,numHid,opt.stcWeights))
net:add(BatchNormalization(numHid, opt.runningVal))
net:add(nn.HardTanh())
net:add(BinarizedNeurons(opt.stcNeurons))
net:add(BinaryLinear(numHid,10,opt.stcWeights))
net:add(nn.BatchNormalization(10))



local dE, param = net:getParameters()
local weight_size = dE:size(1)
local learningRates = torch.Tensor(weight_size):fill(0)
local clipvector = torch.Tensor(weight_size):fill(0)

local counter = 0
for i, layer in ipairs(net.modules) do
   if layer.__typename == 'BinaryLinear' then
      local weight_size = layer.weight:size(1)*layer.weight:size(2)
      local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]+size_w[2]))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+weight_size}}]:fill(1)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
    elseif layer.__typename == 'BatchNormalizationShiftPow2' then
        local weight_size = layer.weight:size(1)
        local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]))
        learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
        clipvector[{{counter+1, counter+weight_size}}]:fill(0)
        counter = counter+weight_size
        local bias_size = layer.bias:size(1)
        learningRates[{{counter+1, counter+bias_size}}]:fill(1)
        clipvector[{{counter+1, counter+bias_size}}]:fill(0)
        counter = counter+bias_size
    elseif layer.__typename == 'nn.BatchNormalization' then
      local weight_size = layer.weight:size(1)
      local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+weight_size}}]:fill(0)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(1)
      clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
  end
end


    -- Add convolutional layers
    ---for i=1,((#args.n_units-1)) do
        -- second convolutional layer
       --- net:add(convLayer(args.n_units[i], args.n_units[i+1],
          ---                  args.filter_size[i+1], args.filter_size[i+1],
          ---                  args.filter_stride[i+1], args.filter_stride[i+1]))
     ---   net:add(args.nl())
  ---  end

  ---  local nel
  ---  if args.gpu >= 0 then
  ---      nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
  ---              :cuda()):nElement()
  ---  else
  ---      nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
  ---  end

    -- reshape all feature planes into a vector per example
 ---   net:add(nn.Reshape(nel))

    -- fully connected layer
---    net:add(nn.Linear(nel, args.n_hid[1]))
---    net:add(args.nl())
---    local last_layer_size = args.n_hid[1]

 ---   for i=1,(#args.n_hid-1) do
        -- add Linear layer
 ---       last_layer_size = args.n_hid[i+1]
 ---       net:add(nn.Linear(args.n_hid[i], last_layer_size))
 ---       net:add(args.nl())
 ---   end

    -- add the last fully connected layer (to actions)
---    net:add(nn.Linear(last_layer_size, args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end
