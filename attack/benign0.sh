#python benign0_func_check.py --target='photodna_nn_cert' --model='/home/yuchen/code/verified_phash/train_verify/saved_models/coco_photodna_ep8/ckpt_best.pth'
#python benign0_func_AUC.py --target='photodna_nn_cert'

python benign0_func_check.py --target='photodna_nn_no_col' --model='/home/yuchen/code/verified_phash/train_verify/saved_models/64_ep1_resv5_l1_aug2_new/ckpt_best.pth'
python benign0_func_AUC.py --target='photodna_nn_no_col'