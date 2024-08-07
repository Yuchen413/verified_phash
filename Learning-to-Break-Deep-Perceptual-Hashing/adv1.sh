#python utils/compute_dataset_hashes.py --model=/home/yuchen/code/verified_phash/Fast-Certified-Robust-Training/64_ep1_resv5_l1_aug2_new_collision/ckpt_best.pth --source=datasets/dog_10K --target=robust

python adv1_collision_attack.py --model=/home/yuchen/code/verified_phash/Fast-Certified-Robust-Training/64_ep1_resv5_l1_aug2_new_collision/ckpt_best.pth --output_folder=collision_test100 --source=/home/yuchen/code/verified_phash/Normal-Training/coco100x100_val --sample_limit=100 --threads=100 --target_hashset=dataset_hashes/dog_10K_hashes_robust.csv

#python adv1_collision_attack.py --model=/home/yuchen/code/verified_phash/Normal-Training/64-coco-hash-resnetv5-l1-aug2-new.pt --output_folder=collision_test100 --source=/home/yuchen/code/verified_phash/Normal-Training/coco100x100_val --sample_limit=100 --threads=100 --target_hashset=dataset_hashes/dog_10K_hashes.csv





#python adv1_collision_attack.py --model=model=/home/yuchen/code/verified_phash/Normal-Training/64-coco-hash-resnetv5-l1-aug2-new-c.pt --output_folder=collision_test1000_c --source=/home/yuchen/code/verified_phash/Normal-Training/coco100x100_val --sample_limit=1000 --threads=100 --target_hashset=dataset_hashes/dog_10K_hashes.csv
#
#python adv1_collision_attack.py --model=/home/yuchen/code/verified_phash/Fast-Certified-Robust-Training/64_ep1_resv5_l1_aug2_new_collision/ckpt_best.pth --output_folder=collision_test1000_c_robust --source=/home/yuchen/code/verified_phash/Normal-Training/coco100x100_val --sample_limit=1000 --threads=100 --target_hashset=dataset_hashes/dog_10K_hashes.csv
