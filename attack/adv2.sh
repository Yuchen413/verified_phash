#python adv2_evasion_attack.py --data=coco --epsilon=0.034 --model=/home/yuchen/code/verified_phash/train_verify/saved_models/coco_photodna_ep0/last_epoch_state_dict.pth --output_folder=evasion_test100_coco --source=/home/yuchen/code/verified_phash/train_verify/data/coco100x100_val --sample_limit=100 --threads=10 --learning_rate=1
#python adv2_evasion_attack.py --data=coco --epsilon=0.034 --model=/home/yuchen/code/verified_phash/train_verify/saved_models/coco_photodna_ep1/ckpt_best.pth --output_folder=evasion_test100_coco --source=/home/yuchen/code/verified_phash/train_verify/data/coco100x100_val --sample_limit=100 --threads=10 --learning_rate=1
python adv2_evasion_attack.py --data=coco --epsilon=0.034 --model=/home/yuchen/code/verified_phash/train_verify/saved_models/base_adv/coco-pdq-ep8-pdg.pt --output_folder=evasion_test100_coco --source=/home/yuchen/code/verified_phash/train_verify/data/coco100x100_val --sample_limit=100 --threads=10 --learning_rate=1

#0.0312
#python adv2_evasion_attack.py --data=mnist --epsilon=0.1 --model=/home/yuchen/code/verified_phash/train_verify/saved_models/mnist_pdq_ep0/last_epoch_state_dict.pth --output_folder=evasion_test100_mnist --source=/home/yuchen/code/verified_phash/train_verify/data/mnist/testing --sample_limit=100 --threads=10 --learning_rate=1e-2
#python adv2_evasion_attack.py --data=mnist --epsilon=0.1 --model=/home/yuchen/code/verified_phash/train_verify/saved_models/mnist_pdq_ep2/ckpt_best.pth --output_folder=evasion_test100_mnist --source=/home/yuchen/code/verified_phash/train_verify/data/mnist/testing --sample_limit=100 --threads=10 --learning_rate=1e-2
#python adv2_evasion_attack.py --data=mnist --epsilon=0.1 --model=/home/yuchen/code/verified_phash/train_verify/saved_models/base_adv/mnist-pdq-ep8-pdg.pt --output_folder=evasion_test100_mnist --source=/home/yuchen/code/verified_phash/train_verify/data/mnist/testing --sample_limit=100 --threads=10 --learning_rate=1e-2



