import logging
import random
random.seed(2017)
import os
import sys

level = logging.NOTSET
log_filename = '%s.log' % __file__
format = '%(asctime)-12s[%(levelname)s] %(message)s'
datefmt='%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=level,
                    format=format,
                    filename=log_filename,
                    datefmt= datefmt)

def oulp_prepare(ids_filename, base_datapath, target_filename):
    data_path = '%s/OULP-C1V1_NormalizedSilhouette(88x128)/' % base_datapath
    csv_path = '%s/OULP-C1V1_SubjectIDList(FormatVersion1.0)/' % base_datapath
    data_id_set = set()
    for line in open(ids_filename):
        data_id = line.strip()
        data_id_set.add(data_id)
    count = len(data_id_set)
    csv_name_list = \
        [
        'IDList_OULP-C1V1-A-55_Gallery.csv'
        ,'IDList_OULP-C1V1-A-55_Probe.csv'
        ,'IDList_OULP-C1V1-A-65_Gallery.csv'
        ,'IDList_OULP-C1V1-A-65_Probe.csv'
        ,'IDList_OULP-C1V1-A-75_Gallery.csv'
        ,'IDList_OULP-C1V1-A-75_Probe.csv'
        ,'IDList_OULP-C1V1-A-85_Gallery.csv'
        ,'IDList_OULP-C1V1-A-85_Probe.csv'
                    ]
    write_lines = []
    for csv_name in csv_name_list:
        csv_filename = '%s/%s' % (csv_path, csv_name)
        for line in open(csv_filename):
            split_line = line.strip().split(',')
            data_id = split_line[0]
            seq_name = 'Seq0%s' % split_line[1]
            start = int(split_line[2])
            end = int(split_line[3])
            if data_id not in data_id_set:
                logging.info('ignore %s' % data_id)
                continue
            else:
                logging.info('proces %s' % data_id)
            target_dirname = '%s/%s/%s/' % (data_path, seq_name, data_id)
            img_names = []
            for i in range(start, end+1):
                img_name = '%s/%08d.png' % (target_dirname, i)
                if not os.path.exists(img_name):
                    logging.error('%s do not exists' % img_name)
                else:
                    img_names.append(img_name)
            line = '%s-%s,%s' % (data_id, csv_name, ','.join(img_names))
            write_lines.append(line)


    random.shuffle(write_lines)
    assert len(write_lines) == count * 8
    target_file = open(target_filename, 'w')
    for line in write_lines:
        target_file.write('%s\n' % line)
    target_file.close()

def oulp_prepare_main(data_path):
    base_path = '%s/../' % data_path
    train_filename = '%sOULP_setting/list_train.txt' % base_path
    test_filename = '%sOULP_setting/list_test.txt' % base_path
    val_filename = '%sOULP_setting/list_val.txt' % base_path
    base_datapath = '%sOULP_C1V1_Pack/' % base_path
    train_target = '%s/oulp_train_data.txt' % data_path
    test_target = '%s/oulp_test_data.txt' % data_path
    val_target = '%s/oulp_val_data.txt' % data_path
    oulp_prepare(train_filename, base_datapath, train_target)
    oulp_prepare(test_filename, base_datapath, test_target)
    oulp_prepare(val_filename, base_datapath, val_target)

if __name__ == '__main__':
    # data path that contains oulp_train_data.txt, oulp_test_data.txt and oulp_val_data.txt
    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    oulp_prepare_main(data_path)
