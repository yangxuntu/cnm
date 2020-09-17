# cnm
1.Prepare the training dataset as in https://github.com/ruotianluo/self-critical.pytorch

2.The training code is:

python train_rs3_lr_new.py --id r78 --caption_model mcap_rs3  --mtopdown_num 2 --mtopdown_res 1 --topdown_res 1 --input_json data/cocobu.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_attr_dir data/cocobu_att --input_rela_dir data/cocobu_att  --input_label_h5 data/cocobu_label.h5 --batch_size 10 --accumulate_number 10 --learning_rate_decay_start 0 --learning_rate 5e-4 --learning_rate_decay_every 5 --learning_rate_rl 1e-4 --learning_rate_decay_every_rl 5 --scheduled_sampling_start 0 --checkpoint_path r78 --save_checkpoint_every 5000 --val_images_use 50 --max_epochs 100 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 37 --train_split train --gpu 0 --combine_att concat --step2_train_after 20 --cont_ver 3 --sg_net_index 0 --relu_mod leaky_relu

3.The evaluation code is:

python eval_rs3.py --dump_images 0 --num_images 5000 --model r78/modelr78${model}.pth --infos_path r78/infos_r78${model}.pkl --language_eval 1 --beam_size 5 --split test --index_eval 1 --gpu 1 --batch_size 100
