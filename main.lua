require 'torch'
require 'nn'
require 'nnx'
require 'optim'

require 'image'
require 'paths'
require 'rnn'

cmd = torch.CmdLine()
cmd:option('-iteration', 100000,'how many iteration')
cmd:option('-gradclip',5,'magnitude of clip on the RNN gradient')
cmd:option('-modelname','wuzifeng','wuzifeng model name you want to load')
cmd:option('-dropout',0.0,'fraction of dropout to use between layers')
cmd:option('-seed',1,'random seed')
cmd:option('-learningrate',1e-3)
cmd:option('-momentum',0.9)
cmd:option('-datapath','/Volumes/Passport/data/gait-rnn', 'base data path')
cmd:option('-calprecision', 2, 'calculate loss on validation every batch')
cmd:option('-calval', 2, 'calculate loss on validation every batch')
cmd:option('-batchsize', 2, 'how many intance in a traning batch')
cmd:option('-loadmodel', '', 'load fullmodel, rnn model, cnn model')
cmd:option('-gpu', false, 'use GPU')
cmd:option('-gpudevice', 1, 'set gpu device')
cmd:option('-mode', 'train', 'train or evaluate')
cmd:option('-datapart', 'test', 'train, val, test')
cmd:option('-debug', false, 'debug? this will output more information which will slow the program')
opt = cmd:parse(arg)
print(opt)

require 'buildModel'
require 'train'
require 'test'
require 'tool'

local prepDataset = require 'prepareDataset'

-- set the GPU
if opt.gpu then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opt.gpudevice)
end

torch.manualSeed(opt.seed)
if opt.gpu then
    cutorch.manualSeed(opt.seed)
end

local dataset = prepDataset.prepareDatasetOULP(opt.datapath, opt.modelname)

for i, item in ipairs({'train', 'val', 'test'}) do
    local item_count = dataset[item]._item_count
    local uniq_count = dataset[item]._uniq_item_count
    info('train data instances %05d, uniq  %04d', item_count, uniq_count)
end

local model, crit

-- build the model
if opt.modelname == 'wuzifeng' then
    model, crit = model_wuzifeng(opt.gpu, opt.dropout)
end
print(model)
if opt.mode == 'train' then
    train_pair(model, crit, dataset)
elseif opt.mode == 'evaluate' then
    model = torch.load(opt.loadmodel)
    model:evaluate()
    info('loaded model from %s', opt.loadmodel)
    local same, diff, prec = evaluate_oulp_simi(dataset[opt.datapart], model)
    info('same, %0.2f, diff, %0.2f, precision, %0.4f', same, diff, prec)
end
