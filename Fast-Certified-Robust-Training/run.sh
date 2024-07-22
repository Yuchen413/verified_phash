python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0039 --dir=64_ep1_resv5_l1_aug2_new_collision_1 --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160  --model='resnet_v5' --lr=5e-4

#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0078 --dir=64_ep2_resv5_l1_aug2_new_collision_1 --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160  --model='resnet_v5' --lr=5e-4

#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0156 --dir=64_ep4_resv5_l1_aug2_new_collision_1 --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160  --model='resnet_v5' --lr=5e-4

#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0312 --dir=64_ep8_resv5_l1_aug2_new_collision --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160  --model='resnet_v5' --lr=5e-4

#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0039 --dir=64_ep1_resv5_l1_aug2 --scheduler_opts=start=1,length=10 --lr-decay-milestones=10,20 --lr-decay-factor=0.5 --num-epochs=40  --model='resnet_v5' --lr=0.01

#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0078 --dir=64_ep2_resv5_l1_aug2 --scheduler_opts=start=1,length=10 --lr-decay-milestones=10,20 --lr-decay-factor=0.5 --num-epochs=40  --model='resnet_v5' --lr=0.01
#
#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0156 --dir=64_ep4_resv5_l1 --scheduler_opts=start=1,length=10 --lr-decay-milestones=10,20 --lr-decay-factor=0.5 --num-epochs=40  --model='resnet_v5' --lr=0.01
#
#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0312 --dir=64_ep8_resv5_l1 --scheduler_opts=start=1,length=10 --lr-decay-milestones=10,20 --lr-decay-factor=0.5 --num-epochs=40  --model='resnet_v5' --lr=0.01

#python train.py --method=fast --eps=0.0078 --dir=64_ep2_resv5_l1 --scheduler_opts=start=1,length=10 --lr-decay-milestones=10,20 --num-epochs=30 --config=config/coco.crown-ibp.json --model='resnet_v5' --lr=0.01
#python train.py --method=fast --eps=0.0156 --dir=64_ep4_resv5_l1 --scheduler_opts=start=1,length=5 --lr-decay-milestones=5,15 --num-epochs=30 --config=config/coco.crown-ibp.json --model='resnet_v5' --lr=0.01
#python train.py --method=fast --eps=0.0312 --dir=64_ep8_resv5_l1 --scheduler_opts=start=1,length=5 --lr-decay-milestones=5,15 --num-epochs=30 --config=config/coco.crown-ibp.json --model='resnet_v5' --lr=0.01