require 'torch'
require 'nn'
require 'nnx'
require 'optim'

require 'image'
require 'paths'
require 'rnn'

require 'tool'

if opt.gpu then
    require 'cunn'
    require 'cutorch'
end

local prepDataset = require 'prepareDataset'


-- train the model on the given dataset
function train_pair(model, criterion, dataset)
    local bsize = opt.batchsize
    local calval = opt.calval
    local calpre = opt.calprecision
    local iteration = opt.iteration

    model:training()
    local max_val_precision = 0

    local min_train_loss = 10
    local min_val_loss = 10
    local min_test_loss = 10

    local parameters, gradParameters = model:getParameters()
    info('Number of parameters:%d', parameters:size(1))
    local optim_state = {
        learningRate = opt.learningrate,
        momentum = opt.momentum,
    }
    local timer = torch.Timer()
    local i = 1
    while i < iteration do
        collectgarbage()
        if ((i - 1) / bsize) % calval == 0 then
            model:evaluate()
            local val_in, val_tar = dataset['val']:next_batch(bsize)
            local val_loss = cal_loss(model, criterion, val_in, val_tar)
            info('%05dth/%05d Val Error %0.6f', i, iteration, val_loss)

            local tes_in, tes_tar = dataset['test']:next_batch(bsize)
            local loss = cal_loss(model, criterion, tes_in, tes_tar)
            info('%05dth/%05d Tes Error %0.6f', i, iteration, loss)
            model:training()
        end


        -- note that due to a problem with SuperCriterion we must cast
        -- from CUDA to double and back before passing data to/from the
        -- criteiron layer - may be fixed in a future update of Torch...
        -- ... or maybe I'm just not using it right!
        local inputs, targets = dataset['train']:next_batch(bsize)
        local feval = function(x)
            batcherror = 0.0
            if x ~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()
            for i, input in ipairs(inputs) do
                --forward
                local output = model:forward(input)
                if opt.gpu then
                    output = convertToDouble(output)
                end
                local target_tensor
                if opt.gpu then
                    target_tensor = torch.CudaTensor(1)
                else
                    target_tensor = torch.Tensor(1)
                end
                target_tensor[1] = targets[i]
                batcherror = batcherror + criterion:forward(output, target_tensor)

                --backward
                local grad = criterion:backward(output, target_tensor)
                if opt.gpu then
                    grad = convertToCuda(grad)
                end
                model:backward(input, grad)
                gradParameters:clamp(-opt.gradclip, opt.gradclip)
            end
            batcherror = batcherror / #inputs
            gradParameters:div(#inputs)
            return batcherror, gradParameters
        end
        optim.sgd(feval, parameters, optim_state)

        local time = timer:time().real
        timer:reset()
        info('%05dth/%05d Tra Error %0.6f, %d'
            , i, iteration, batcherror, time)
        if batcherror  < min_train_loss then
            min_train_loss = batcherror
            name = string.format('%s_tra_%0.04f_i%04d'
                , opt.modelname, min_train_loss, i)
            save_model(model, name)
        end

        if i > 1 and ((i - 1) / bsize) % calpre == 0 then
            model:evaluate()
            local avg_same, avg_diff, avg_precision
            local same, diff, prec = evaluate_oulp_simi(dataset['val'], model)
            if prec > max_val_precision then
                info('change max precision from %0.2f to %0.2f'
                                            , max_val_precision, prec)
                max_val_precision = prec
                local name = string.format('%s_valpre_%0.04f_i%04d'
                                , opt.modelname, max_val_precision, i)
                save_model(model, name)
            else
                info('do not change max_precision from %0.2f to %0.2f'
                            , max_val_precision, prec)
            end
        end
        i = i + bsize
    end
    return model
end
