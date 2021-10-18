# cnm
1.Prepare the training dataset as in https://github.com/ruotianluo/self-critical.pytorch

2.The training code is:

python train_rs3_lr_new.py --id r78 --caption_model mcap_rs3  --mtopdown_num 2 --mtopdown_res 1 --topdown_res 1 --input_json data/cocobu.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_attr_dir data/cocobu_att --input_rela_dir data/cocobu_att  --input_label_h5 data/cocobu_label.h5 --batch_size 10 --accumulate_number 10 --learning_rate_decay_start 0 --learning_rate 5e-4 --learning_rate_decay_every 5 --learning_rate_rl 1e-4 --learning_rate_decay_every_rl 5 --scheduled_sampling_start 0 --checkpoint_path r78 --save_checkpoint_every 5000 --val_images_use 50 --max_epochs 100 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 37 --train_split train --gpu 0 --combine_att concat --step2_train_after 20 --cont_ver 3 --sg_net_index 0 --relu_mod leaky_relu

3.The evaluation code is:

python eval_rs3.py --dump_images 0 --num_images 5000 --model r78/modelr78${model}.pth --infos_path r78/infos_r78${model}.pkl --language_eval 1 --beam_size 5 --split test --index_eval 1 --gpu 1 --batch_size 100

CVLNM/ pytorch 0.4.0
I provide the anaconda environment for running my code in https://drive.google.com/drive/folders/1GvwpchUnfqUjvlpWTYbmEvhvkJTIWWRb?usp=sharing. You should download the file ''environment_yx1.yml'' from this link and set up the environment as follows.
1.Download the anaconda from the website https://www.anaconda.com/ and install it.
2.Go to website https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html?highlight=environment to learn how to 'creating an environment from an environment.yml file'.
```
conda env create -f environment_yx1.yml
```
3.After installing anaconda and setting up the environment, run the following code to get into the environment.
```
source activate yx1
```
If you want to exit from this environment, you can run the following code to exit.
```
source deactivate
```
# Download Bottom-up features.
Download pre-extracted feature from https://github.com/peteanderson80/bottom-up-attention. You can either download adaptive one or fixed one. We use the ''10 to 100 features per image (adaptive)'' in our experiments.
For example:
```
mkdir data/bu_data; cd data/bu_data
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip
```
Then :
```
python script/make_bu_data.py --output_dir data/cocobu
```
This will create data/cocobu_fc, data/cocobu_att and data/cocobu_box. 
# Training the model
1.After downloading the codes and meta data, you can train the model by using the following code:
```
