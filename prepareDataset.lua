require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'paths'
require 'image'

if opt.gpu then
    require 'cunn'
    require 'cutorch'
end


prepareDataset = {}
local DatasetGenerator = {}
DatasetGenerator.__index = DatasetGenerator
setmetatable(DatasetGenerator, {
    __call = function(cls, ...)
        return cls.new(...)
    end,
})


function DatasetGenerator.new(filename, video_id_image, hids, is_grey, modelname)
    local self = setmetatable({}, DatasetGenerator)
    if is_grey == nil then
        is_grey = false
    end
    self._is_grey = is_grey
    self._data = { {}, {} }
    self._video_images = {}
    self._pos_index = 1
    self._neg_index = 1
    self._next_item_batch_idx = 1
    self._next_gei_batch_idx = 1
    self._size = 0
    self._hids = hids

    self._item_count = 0
    self._uniq_item_to_number = {}
    self._number_to_uniq_item = {}
    self._uniq_item_count = 0
    self._item_data = {}
    self._modelname = modelname
    local to_number_start = 1
    if path.exists(filename) then
        for line in io.lines(filename) do
            local res = splitByComma(line)
            local is_pos = tonumber(res[3])
            if is_pos == 0 then
                is_pos = 2
            end
            table.insert(self._data[is_pos], { res[1], res[2] })
        end
    end
    self._size = #self._data[1]
    for line in io.lines(video_id_image) do
        self._item_count = self._item_count + 1
        local res = splitByComma(line)
        local item = res[1]
        local item_id_end, _ = string.find(item, '-')
        local item_id = string.sub(item, 1, item_id_end - 1)
        if self._uniq_item_to_number[item_id] == nil then
            self._uniq_item_to_number[item_id] = to_number_start
            table.insert(self._number_to_uniq_item, item_id)
            to_number_start = to_number_start + 1
        end
        self._video_images[res[1]] = {}
        table.insert(self._item_data, item)
        for i = 2, #res do
            table.insert(self._video_images[res[1]], res[i])
        end
    end
    self._uniq_item_count = to_number_start - 1
    return self
end

function DatasetGenerator:set_pos_index(index)
    self._pos_index = index
end

function DatasetGenerator:set_neg_index(index)
    self._neg_index = index
end

function DatasetGenerator:next_gei_pair_softmax(size)
    local targets = {}
    local inputs = {}
    local batch = self:next_gei_pair_batch_oulp(size)
    for i = 1, #batch do
        local video1 = batch[i][1]
        local video2 = batch[i][2]
        local hid1 = batch[i][3]
        local hid2 = batch[i][4]
        local pos = batch[i][5]
        local target
        if pos then
            target = 2
        else
            target = 1
        end
        table.insert(inputs, {video1, video2})
        table.insert(targets, target)
    end
    return inputs, targets
end

function DatasetGenerator:next_seq_pair_softmax(size)
    local targets = {}
    local inputs = {}
    local batch = self:next_seq_pair_batch_oulp(size)
    for i = 1, #batch do
        local video1 = batch[i][1]
        local video2 = batch[i][2]
        local hid1 = batch[i][3]
        local hid2 = batch[i][4]
        local pos = batch[i][5]
        local target
        if pos then
            target = 2
        else
            target = 1
        end
        table.insert(inputs, {video1, video2})
        table.insert(targets, target)
    end
    return inputs, targets
end

function DatasetGenerator:load_gei(video_id)
    local img_paths = self._video_images[video_id]
    local resize = true
    local size = {126, 126}
    if opt.modelName == 'GEINet' then
        resize = false
        size = {128, 88}
    end
    local video_images = self:load_sequence_imgs(img_paths, resize, size)
    local res = video_images[1]
    for i=2,#video_images do
      res = torch.add(res, video_images[i])
    end
    
    local gei = torch.div(res, #video_images)
    return gei
end


function DatasetGenerator:_load_image_paths(video_id, image_count)
    local origin_count = #self._video_images[video_id]
    image_count = opt.imgup
    if image_count <= 0 or image_count > origin_count then
        image_count = origin_count
    end
    local image_start = 1
    local images = {}
    local j = image_start
    while j <= image_start + image_count - 1 do
        local image_path_idx = (j - 1) % origin_count + 1
        table.insert(images, self._video_images[video_id][image_path_idx])
        j = j + opt.imgsample
    end
    return images
end

function DatasetGenerator:load_images(video_id, image_count, resize, size)
    if resize == nil then
        resize = false
    end
    local images = self:_load_image_paths(video_id, image_count)
    local video_images = self:load_sequence_imgs(images, resize, size)
    return video_images
end

--load all images into a flat list
function DatasetGenerator:load_sequence_imgs(filesList, resize, size, diff_dim)
    local nImgs = #filesList
    local dim = 3
    local height = 56
    local width = 40
    if resize and size ~= nil then
      height = size[1]
      width = size[2]
    end

    local image_pixel_data_tbl = {}
    if #filesList == 0 then
        info(string.format('no image found'))
    end

    local previous_img = {}
    for i, filename in ipairs(filesList) do
        local img = image.load(filename, 3)
        if resize then
          img = image.scale(img, width, height)
        end
        -- --allocate storage
        --img = image.rgb2lab(img):type('torch.DoubleTensor')
        if self._is_grey then
            img = image.rgb2y(img):type('torch.DoubleTensor')
        end
        if opt.gpu then
            img = img:cuda()
        end
        table.insert(previous_img, img)
    end
    return previous_img
end

function DatasetGenerator:reset_next_item_batch_idx()
    self._next_item_batch_idx = 1
end


function DatasetGenerator:next_gei_batch(batch_size)
    local i = self._next_gei_batch_idx
    local res = {}
    while #res ~= batch_size do
        if i > #self._item_data then
            info('reset self._next_gei_batch_idx to 1')
            i = 1
        end
        local item = self._item_data[i]
        local item_id_end, _ = string.find(item, '-')
        local item_id = string.sub(item, 1, item_id_end - 1)
        local item_id_number = self._uniq_item_to_number[item_id]
        local images = self:load_gei(item)
        table.insert(res, { images, item_id_number })
        i = i + 1
    end
    self._next_gei_batch_idx = i
    return res
end

function DatasetGenerator:_get_item_id_number(item)
    local item_id_end, _ = string.find(item, '-')
    local item_id = string.sub(item, 1, item_id_end - 1)
    local item_id_number = self._uniq_item_to_number[item_id]
    return item_id_number
end

function DatasetGenerator:_next_pair_batch_oulp_item(is_pos)
    local seqs = {'Probe', 'Gallery' }
    local views = {55, 65, 75, 85 }
    local item_tpl = "%s-IDList_OULP-C1V1-A-%s_%s.csv"
    local item1, item2
    if is_pos then
        local hid_idx = torch.random(1, self._uniq_item_count)
        local s_i1 = torch.random(1, #seqs)
        local s_i2 = torch.random(1, #seqs - 1)

        local v_i1 = torch.random(1, #views)
        local v_i2 = torch.random(1, #views - 1)

        if s_i2 >= s_i1 then
            s_i2 = s_i2 + 1
        end

        if v_i2 >= v_i1 then
            v_i2 = v_i1 + 1
        end

        local hid = self._number_to_uniq_item[hid_idx]
        item1 = string.format(item_tpl, hid, views[v_i1], seqs[s_i1])
        item2 = string.format(item_tpl, hid, views[v_i2], seqs[s_i2])
    else
        local hid1_idx = torch.random(1, self._uniq_item_count)
        local hid2_idx = torch.random(1, self._uniq_item_count - 1)
        if hid2_idx >= hid1_idx then
            hid2_idx = hid2_idx + 1
        end
        local s_i1 = torch.random(1, #seqs)
        local s_i2 = torch.random(1, #seqs)
        local v_i1 = torch.random(1, #views)
        local v_i2 = torch.random(1, #views)
        local hid1 = self._number_to_uniq_item[hid1_idx]
        local hid2 = self._number_to_uniq_item[hid2_idx]
        item1 = string.format(item_tpl, hid1, views[v_i1], seqs[s_i1])
        item2 = string.format(item_tpl, hid2, views[v_i2], seqs[s_i2])
    end
    return item1, item2

end

function DatasetGenerator:next_gei_pair_batch_oulp(batch_size)
    local pos = true
    local pos_batch = self:_next_gei_pair_batch_oulp(batch_size/2, pos)
    local neg_batch = self:_next_gei_pair_batch_oulp(batch_size/2, not pos)
    local res = {}
    for i=1, #pos_batch do
        table.insert(res, pos_batch[i])
        table.insert(res, neg_batch[i])
    end
    return res
end


function DatasetGenerator:_next_gei_pair_batch_oulp(batch_size, is_pos)
    local res = {}
    local item1, item2
    while #res ~= batch_size do
        item1, item2 = self:_next_pair_batch_oulp_item(is_pos)
        local item1_number = self:_get_item_id_number(item1)
        local item2_number = self:_get_item_id_number(item2)
        local images1 = self:load_gei(item1)
        local images2 = self:load_gei(item2)
        table.insert(res, {images1, images2, item1_number, item2_number, is_pos})
    end
    return res
end

function DatasetGenerator:next_batch(batch_size)
    if self._modelname == 'wuzifeng' then
        return self:next_gei_pair_softmax(batch_size)
    end
end

function DatasetGenerator:load_item(item)
    if self._modelname == 'wuzifeng' then
        return self:load_gei(item)
    end
end

function DatasetGenerator:size()
    return self._size;
end


function prepareDataset.prepareDatasetOULP(datapath, modelname)
    local train_filename = string.format('%s/oulp_train_data.txt', datapath)
    local test_filename = string.format('%s/oulp_test_data.txt', datapath)
    local val_filename = string.format('%s/oulp_val_data.txt', datapath)
    local res = {}
    local is_grey = true
    res['train'] = DatasetGenerator.new('', train_filename, '', is_grey, modelname)
    res['val'] = DatasetGenerator.new('', val_filename, '', is_grey, modelname)
    res['test'] = DatasetGenerator.new('', test_filename, '', is_grey, modelname)
    info(string.format('load data from %s, %s, %s', train_filename, val_filename, test_filename))
    return res
end
return prepareDataset
