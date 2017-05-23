function model_wuzifeng(gpu, dropout)
    local input_h = 126
    local input_w = 126
    local input_channel = 1
    local conv_kernel = { 16, 64, 256 }
    local conv_size = { 7, 7, 7 }
    local conv_tride = { 1, 1, 1 }
    local pool_size = { 2, 2 }
    local pool_tride = { 2, 2 }
    local LRN_size = 5
    local LRN_alpha = 0.0001
    local LRN_beta = 0.75
    local LRN_k = 2

    local c1 = nn.Sequential()
    if gpu then
        c1:add(nn.SpatialConvolutionMM(input_channel, conv_kernel[1], conv_size[1], conv_size[1], conv_tride[1], conv_tride[1]))
    else
        c1:add(nn.SpatialConvolution(input_channel, conv_kernel[1], conv_size[1], conv_size[1], conv_tride[1], conv_tride[1]))
    end
    c1:add(nn.ReLU())
    c1:add(nn.SpatialCrossMapLRN(LRN_size, LRN_alpha, LRN_beta, LRN_k))
    c1:add(nn.SpatialMaxPooling(pool_size[1], pool_size[1], pool_tride[1], pool_tride[1]))

    if gpu then
        c1:add(nn.SpatialConvolutionMM(conv_kernel[1], conv_kernel[2], conv_size[2], conv_size[2], conv_tride[2], conv_tride[2]))
    else
        c1:add(nn.SpatialConvolution(conv_kernel[1], conv_kernel[2], conv_size[2], conv_size[2], conv_tride[2], conv_tride[2]))
    end
    c1:add(nn.ReLU())
    c1:add(nn.SpatialCrossMapLRN(LRN_size, LRN_alpha, LRN_beta, LRN_k))
    c1:add(nn.SpatialMaxPooling(pool_size[2], pool_size[2], pool_tride[2], pool_tride[2]))


    if gpu then
        c1:cuda()
    end
    local c2 = c1:clone('weight', 'bias', 'gradWeight', 'gradBias')

    local two = nn.ParallelTable()
    two:add(c1)
    two:add(c2)
    if gpu then
        two:cuda()
    end

    local merge = nn.Sequential()
    merge:add(nn.CSubTable())
    merge:add(nn.Abs())


    local c3 = nn.Sequential()
    if gpu then
        c3:add(nn.SpatialConvolutionMM(conv_kernel[2], conv_kernel[3], conv_size[3], conv_size[3], conv_tride[3], conv_tride[3]))
    else
        c3:add(nn.SpatialConvolution(conv_kernel[2], conv_kernel[3], conv_size[3], conv_size[3], conv_tride[3], conv_tride[3]))
    end
    local fully = conv_kernel[3] * 21 * 21
    c3:add(nn.Reshape(1, fully))
    c3:add(nn.Dropout(dropout))
    c3:add(nn.Linear(fully, 2))
    c3:add(nn.LogSoftMax())

    local model = nn.Sequential()
    model:add(two)
    model:add(merge)
    model:add(c3)
    if gpu then
        model:cuda()
    end
    local crit = nn.ClassNLLCriterion()
    return model, crit
end
