require 'torch'
require 'image' 

function print_log(msg, level)
    print(string.format('%s[%s] %s', os.date('%Y-%m-%d %X'), level, msg))
end

function info(msg_tpl, ...)
    local arg = {...}
    print_log(string.format(msg_tpl, unpack(arg)), 'INFO')
end

function splitByComma(str)
    local res = {}
    for word in string.gmatch(str, '([^,\n]+)') do
        table.insert(res, word)
    end
    return res
end

function convertToType(output, c_type)
    if type(output) == 'table' then
        for i=1, #output do
            if c_type == 'double' then
                output[i] = output[i]:double()
            elseif c_type == 'cuda' then
                output[i] = output[i]:cuda()
            end
        end
    else
        if c_type == 'double' then
            output = output:double()
        elseif c_type == 'cuda' then
            output = output:cuda()
        end
    end
    return output

end

function convertToDouble(output)
    return convertToType(output, 'double')
end

function convertToCuda(output)
    return convertToType(output, 'cuda')
end

function prepareTensorHolder(batch_size, video_len, dim, height, width, gpu, two)
  local h = height
  local w = width
  local inputs = {}
  for i=1, batch_size do
    local x = {}
    local y = {}
    for t = 1, video_len do
        if gpu then
            table.insert(x, torch.zeros(dim, h, w):cuda())
            table.insert(y, torch.zeros(dim, h, w):cuda())
        else
            table.insert(x, torch.zeros(dim, h, w))
            table.insert(y, torch.zeros(dim, h, w))
        end
    end
    local input
    if two then
        input = { x, y }
    else
        input = x
    end
    table.insert(inputs, input)
  end
  return inputs
end

function _cal_loss_single_match(model, crit, input, target) 
    local align_tbl = align_seqs(model, input)
    local min_dist = model:forward(align_tbl)
    local b = torch.Tensor(1)
    b[1] = target
    -- print('min_dist and target', min_dist[1], target)
    return crit:forward(min_dist, b)
end

function cal_loss(model, crit, inputs, targets)
    local loss = 0
    for i, input in ipairs(inputs) do
        local l = 0
        if opt.modelname == 'singlematch' then
            l = _cal_loss_single_match(model, crit, inputs[i], targets[i])
        else
            local output = model:forward(input)
            output = convertToDouble(output)
            l = crit:forward(output, targets[i])
        end
        loss = loss + l
    end
    loss = loss / #inputs
    return loss
end

function save_model(model, name)
    local dirname = './trainedNets'
    os.execute("mkdir  -p " .. dirname)
    local filename = string.format('%s/%s.t7', dirname, name)
    torch.save(filename, model)
end

function feature_extract_model(model, modelname)
    if modelname == 'wuzifeng' then
        return model:get(1):get(1)
    end
end

function merge_model(model, modelname)
    local ret = nn.Sequential()
    if modelname == 'wuzifeng' then
      ret:add(model:get(2))
      ret:add(model:get(3))
    end
    return ret
end
  
function clone_value(val)
  if type(val) == 'table' then
    local ret = {}
    for i=1,#val do
      table.insert(ret, val[i]:clone())
    end
    return ret
  else
    return val:clone()
  end
end

function cal_eer(pro, res)
    -- calculate equal error rate
    -- frr for false reject rate
    -- far for false accept rate
    local frr_tbl = {}
    local far_tbl = {}
    if #pro ~= #res then
        error('length of pro and res are different')
    end
    local f
    local true_count = 0
    local false_count = 0
    for i=1,#res do
        if res[i] then
            true_count = true_count + 1
        end
    end
    false_count = #res - true_count
    for theta=0.000, 1, 0.001 do
        local frr_count = 0
        local far_count = 0
        for i=1,#pro do
            if pro[i] > theta and not res[i] then
                far_count = far_count + 1
            end

            if pro[i] < theta and res[i] then
                frr_count = frr_count + 1
            end
        end
        local frr = frr_count * 1.0 / true_count
        table.insert(frr_tbl, frr)

        local far = far_count * 1.0 / false_count
        table.insert(far_tbl, far)
    end
    local eer = 1
    local diff = 1
    for i=1,#frr_tbl do
        local tmp_diff = math.abs(frr_tbl[i] - far_tbl[i])
        if tmp_diff < diff then
            diff = tmp_diff
            eer = (frr_tbl[i] + far_tbl[i]) / 2.0
        end
    end
    return eer
end

function put2one(feature_maps)
    local map_count = feature_maps:size(1)
    local height = feature_maps:size(2)
    local mul = torch.sqrt(map_count)
    if mul - math.floor(mul) > 0.0001 then
        error('map count is not sqrtable, ' ..map_count)
    end
    local res = torch.Tensor(1, mul*height, mul*height)
    for i=1,map_count do
        local start_h = 1 + math.floor((i-1)/mul) * height
        local start_w = 1 + (i - math.floor((i-1)/mul)*mul - 1) * height
        for h=1,height do
            for w=1,height do
                local t_h = start_h + h - 1
                local t_w = start_w + w - 1
                res[1][t_h][t_w] = feature_maps[i][h][w]
            end
        end
    end
    return res
end

function diff2img_withorder(img1, img2)
    local res = img2 - img1
    local clip_zero = (res + torch.abs(res:clone())) / 2
    return clip_zero
end

function read_gray_img(filename, height, width)
    local img = image.load(filename, 3)
    img = image.scale(img, width, height)
    img = image.rgb2y(img):type('torch.DoubleTensor')
    return img
end

function diff2img_withorder_demo()
    cmd = torch.CmdLine()
    cmd:option('-img1', '', '')
    cmd:option('-img2', '', '')
    cmd:option('-savefilename', '', '')
    opt = cmd:parse(arg)
    print(opt)
    local width = 126
    local height = 126
    local img1 = read_gray_img(opt.img1, height, width)
    local img2 = read_gray_img(opt.img2, height, width)
    local res = diff2img_withorder(img1, img2)
    image.save(opt.savefilename, res)
end
