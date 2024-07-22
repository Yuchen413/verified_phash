import re
import matplotlib.pyplot as plt

log_data_fast = '''
INFO     2024-05-01 14:27:17,900 Epoch 1, learning rate [0.01], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:27:53,498 [ 1]: eps=0.00000000 active=0.3864 inactive=0.6084 CE=0.0658 Rob_Loss=4161684.7531 Rob_Err=0.5580 L_tightness=0.4859 L_relu=0.0082 L_std=0.0606 loss_reg=0.4941 Loss=0.3129 grad_norm=0.2569 wnorm=49.0322 Time=0.0323
INFO     2024-05-01 14:27:53,500 Epoch time: 35.5991, Total time: 35.5991
INFO     2024-05-01 14:27:53,500 Test without loss fusion
INFO     2024-05-01 14:27:55,961 [ 1]: eps=0.00000000 active=0.3934 inactive=0.6066 CE=0.0178 Rob_Loss=0.0179 Rob_Err=0.0366 L_tightness=0.1282 L_relu=0.0004 L_std=0.1147 loss_reg=0.1286 Loss=0.0821 wnorm=49.2160 Time=0.0111
INFO     2024-05-01 14:27:55,975 
INFO     2024-05-01 14:27:55,975 Epoch 2, learning rate [0.01], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:28:31,513 [ 2]: eps=0.00000000 active=0.3966 inactive=0.6034 CE=0.0154 Rob_Loss=0.0155 Rob_Err=0.0309 L_tightness=0.0224 L_relu=0.0000 L_std=0.1635 loss_reg=0.0224 Loss=0.0266 grad_norm=0.0648 wnorm=49.1557 Time=0.0323
INFO     2024-05-01 14:28:31,515 Epoch time: 35.5402, Total time: 71.1393
INFO     2024-05-01 14:28:31,515 Test without loss fusion
INFO     2024-05-01 14:28:33,976 [ 2]: eps=0.00000000 active=0.4030 inactive=0.5970 CE=0.0124 Rob_Loss=0.0124 Rob_Err=0.0243 L_tightness=0.0000 L_relu=0.0000 L_std=0.1747 loss_reg=0.0000 Loss=0.0124 wnorm=49.2720 Time=0.0111
INFO     2024-05-01 14:28:34,023 
INFO     2024-05-01 14:28:34,023 Epoch 3, learning rate [0.01], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:29:09,644 [ 3]: eps=0.00000000 active=0.4027 inactive=0.5973 CE=0.0119 Rob_Loss=0.0120 Rob_Err=0.0232 L_tightness=0.0000 L_relu=0.0000 L_std=0.1863 loss_reg=0.0000 Loss=0.0119 grad_norm=0.0245 wnorm=49.3137 Time=0.0323
INFO     2024-05-01 14:29:09,646 Epoch time: 35.6222, Total time: 106.7615
INFO     2024-05-01 14:29:09,646 Test without loss fusion
INFO     2024-05-01 14:29:12,111 [ 3]: eps=0.00000000 active=0.4002 inactive=0.5998 CE=0.0115 Rob_Loss=0.0115 Rob_Err=0.0224 L_tightness=0.0000 L_relu=0.0000 L_std=0.1908 loss_reg=0.0000 Loss=0.0115 wnorm=49.3842 Time=0.0111
INFO     2024-05-01 14:29:12,156 
INFO     2024-05-01 14:29:12,156 Epoch 4, learning rate [0.01], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:29:47,753 [ 4]: eps=0.00000000 active=0.4079 inactive=0.5921 CE=0.0115 Rob_Loss=0.0115 Rob_Err=0.0224 L_tightness=0.0000 L_relu=0.0003 L_std=0.1919 loss_reg=0.0004 Loss=0.0117 grad_norm=0.0226 wnorm=49.6173 Time=0.0323
INFO     2024-05-01 14:29:47,755 Epoch time: 35.5987, Total time: 142.3602
INFO     2024-05-01 14:29:47,755 Test without loss fusion
INFO     2024-05-01 14:29:50,237 [ 4]: eps=0.00000000 active=0.4350 inactive=0.5650 CE=0.0113 Rob_Loss=0.0113 Rob_Err=0.0219 L_tightness=0.0000 L_relu=0.0000 L_std=0.1600 loss_reg=0.0000 Loss=0.0113 wnorm=50.0811 Time=0.0111
INFO     2024-05-01 14:29:50,286 
INFO     2024-05-01 14:29:50,287 Epoch 5, learning rate [0.01], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:30:26,123 [ 5]: eps=0.00000150 active=0.4433 inactive=0.5567 CE=0.0110 Rob_Loss=0.0110 Rob_Err=0.0213 L_tightness=0.0863 L_relu=0.0000 L_std=0.1713 loss_reg=0.0863 Loss=0.0542 grad_norm=0.0145 wnorm=50.3342 Time=0.0329
INFO     2024-05-01 14:30:26,124 Epoch time: 35.8376, Total time: 178.1978
INFO     2024-05-01 14:30:26,124 Test without loss fusion
INFO     2024-05-01 14:30:28,596 [ 5]: eps=0.00000150 active=0.4492 inactive=0.5508 CE=0.0105 Rob_Loss=0.0106 Rob_Err=0.0204 L_tightness=0.0000 L_relu=0.0000 L_std=0.1832 loss_reg=0.0000 Loss=0.0106 wnorm=50.5606 Time=0.0112
INFO     2024-05-01 14:30:28,642 
INFO     2024-05-01 14:30:28,642 Epoch 6, learning rate [0.002], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:31:04,580 [ 6]: eps=0.00002417 active=0.4452 inactive=0.5548 CE=0.0103 Rob_Loss=0.0109 Rob_Err=0.0200 L_tightness=0.0000 L_relu=0.0000 L_std=0.1864 loss_reg=0.0000 Loss=0.0109 grad_norm=0.0143 wnorm=50.5500 Time=0.0330
INFO     2024-05-01 14:31:04,581 Epoch time: 35.9394, Total time: 214.1372
INFO     2024-05-01 14:31:04,581 Test without loss fusion
INFO     2024-05-01 14:31:07,070 [ 6]: eps=0.00002417 active=0.4380 inactive=0.5619 CE=0.0102 Rob_Loss=0.0118 Rob_Err=0.0200 L_tightness=0.0000 L_relu=0.0000 L_std=0.1867 loss_reg=0.0000 Loss=0.0118 wnorm=50.4803 Time=0.0112
INFO     2024-05-01 14:31:07,115 
INFO     2024-05-01 14:31:07,116 Epoch 7, learning rate [0.002], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:31:43,508 [ 7]: eps=0.00012255 active=0.4270 inactive=0.5729 CE=0.0104 Rob_Loss=0.0129 Rob_Err=0.0204 L_tightness=0.0000 L_relu=0.0000 L_std=0.1852 loss_reg=0.0000 Loss=0.0129 grad_norm=0.0180 wnorm=50.2310 Time=0.0335
INFO     2024-05-01 14:31:43,510 Epoch time: 36.3942, Total time: 250.5314
INFO     2024-05-01 14:31:43,510 Test without loss fusion
INFO     2024-05-01 14:31:45,993 [ 7]: eps=0.00012255 active=0.4200 inactive=0.5798 CE=0.0105 Rob_Loss=0.0125 Rob_Err=0.0206 L_tightness=0.0000 L_relu=0.0000 L_std=0.1838 loss_reg=0.0000 Loss=0.0125 wnorm=49.9752 Time=0.0112
INFO     2024-05-01 14:31:46,039 
INFO     2024-05-01 14:31:46,039 Epoch 8, learning rate [0.002], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:32:22,301 [ 8]: eps=0.00037937 active=0.4153 inactive=0.5844 CE=0.0106 Rob_Loss=0.0122 Rob_Err=0.0208 L_tightness=0.0000 L_relu=0.0000 L_std=0.1802 loss_reg=0.0000 Loss=0.0122 grad_norm=0.0168 wnorm=49.8407 Time=0.0335
INFO     2024-05-01 14:32:22,303 Epoch time: 36.2639, Total time: 286.7953
INFO     2024-05-01 14:32:22,303 Test without loss fusion
INFO     2024-05-01 14:32:24,787 [ 8]: eps=0.00037937 active=0.4095 inactive=0.5901 CE=0.0105 Rob_Loss=0.0116 Rob_Err=0.0207 L_tightness=0.0000 L_relu=0.0000 L_std=0.1771 loss_reg=0.0000 Loss=0.0116 wnorm=49.7654 Time=0.0112
INFO     2024-05-01 14:32:24,832 
INFO     2024-05-01 14:32:24,833 Epoch 9, learning rate [0.002], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:33:01,057 [ 9]: eps=0.00069936 active=0.4052 inactive=0.5943 CE=0.0104 Rob_Loss=0.0113 Rob_Err=0.0203 L_tightness=0.0000 L_relu=0.0000 L_std=0.1767 loss_reg=0.0000 Loss=0.0113 grad_norm=0.0151 wnorm=49.7474 Time=0.0334
INFO     2024-05-01 14:33:01,058 Epoch time: 36.2258, Total time: 323.0211
INFO     2024-05-01 14:33:01,059 Test without loss fusion
INFO     2024-05-01 14:33:03,525 [ 9]: eps=0.00069936 active=0.4002 inactive=0.5992 CE=0.0102 Rob_Loss=0.0110 Rob_Err=0.0199 L_tightness=0.0000 L_relu=0.0000 L_std=0.1742 loss_reg=0.0000 Loss=0.0110 wnorm=49.7438 Time=0.0111
INFO     2024-05-01 14:33:03,571 
INFO     2024-05-01 14:33:03,571 Epoch 10, learning rate [0.002], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:33:40,225 [10]: eps=0.00101935 active=0.3959 inactive=0.6034 CE=0.0101 Rob_Loss=0.0109 Rob_Err=0.0197 L_tightness=0.0000 L_relu=0.0000 L_std=0.1752 loss_reg=0.0000 Loss=0.0109 grad_norm=0.0142 wnorm=49.7538 Time=0.0334
INFO     2024-05-01 14:33:40,227 Epoch time: 36.6557, Total time: 359.6768
INFO     2024-05-01 14:33:40,227 Test without loss fusion
INFO     2024-05-01 14:33:42,689 [10]: eps=0.00101935 active=0.3919 inactive=0.6073 CE=0.0100 Rob_Loss=0.0108 Rob_Err=0.0195 L_tightness=0.0000 L_relu=0.0000 L_std=0.1738 loss_reg=0.0000 Loss=0.0108 wnorm=49.7698 Time=0.0111
INFO     2024-05-01 14:33:42,734 
INFO     2024-05-01 14:33:42,734 Epoch 11, learning rate [0.002], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:34:18,644 [11]: eps=0.00133934 active=0.3870 inactive=0.6121 CE=0.0096 Rob_Loss=0.0105 Rob_Err=0.0187 L_tightness=0.0000 L_relu=0.0000 L_std=0.1749 loss_reg=0.0000 Loss=0.0105 grad_norm=0.0142 wnorm=49.7995 Time=0.0330
INFO     2024-05-01 14:34:18,645 Epoch time: 35.9113, Total time: 395.5880
INFO     2024-05-01 14:34:18,646 Test without loss fusion
INFO     2024-05-01 14:34:21,115 [11]: eps=0.00133934 active=0.3820 inactive=0.6170 CE=0.0093 Rob_Loss=0.0101 Rob_Err=0.0181 L_tightness=0.0000 L_relu=0.0000 L_std=0.1745 loss_reg=0.0000 Loss=0.0101 wnorm=49.8304 Time=0.0112
INFO     2024-05-01 14:34:21,160 
INFO     2024-05-01 14:34:21,160 Epoch 12, learning rate [0.002], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:34:57,091 [12]: eps=0.00165932 active=0.3800 inactive=0.6189 CE=0.0090 Rob_Loss=0.0098 Rob_Err=0.0175 L_tightness=0.0000 L_relu=0.0000 L_std=0.1757 loss_reg=0.0000 Loss=0.0098 grad_norm=0.0134 wnorm=49.8547 Time=0.0330
INFO     2024-05-01 14:34:57,093 Epoch time: 35.9326, Total time: 431.5206
INFO     2024-05-01 14:34:57,093 Test without loss fusion
INFO     2024-05-01 14:34:59,560 [12]: eps=0.00165932 active=0.3774 inactive=0.6214 CE=0.0087 Rob_Loss=0.0094 Rob_Err=0.0170 L_tightness=0.0000 L_relu=0.0000 L_std=0.1795 loss_reg=0.0000 Loss=0.0094 wnorm=49.8769 Time=0.0111
INFO     2024-05-01 14:34:59,606 
INFO     2024-05-01 14:34:59,606 Epoch 13, learning rate [0.002], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:35:35,444 [13]: eps=0.00197931 active=0.3768 inactive=0.6219 CE=0.0085 Rob_Loss=0.0092 Rob_Err=0.0166 L_tightness=0.0000 L_relu=0.0000 L_std=0.1801 loss_reg=0.0000 Loss=0.0092 grad_norm=0.0124 wnorm=49.8944 Time=0.0328
INFO     2024-05-01 14:35:35,446 Epoch time: 35.8399, Total time: 467.3606
INFO     2024-05-01 14:35:35,446 Test without loss fusion
INFO     2024-05-01 14:35:37,909 [13]: eps=0.00197931 active=0.3768 inactive=0.6219 CE=0.0083 Rob_Loss=0.0090 Rob_Err=0.0162 L_tightness=0.0000 L_relu=0.0000 L_std=0.1805 loss_reg=0.0000 Loss=0.0090 wnorm=49.9130 Time=0.0111
INFO     2024-05-01 14:35:37,954 
INFO     2024-05-01 14:35:37,955 Epoch 14, learning rate [0.002], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:36:14,054 [14]: eps=0.00229930 active=0.3747 inactive=0.6239 CE=0.0081 Rob_Loss=0.0088 Rob_Err=0.0159 L_tightness=0.0000 L_relu=0.0000 L_std=0.1847 loss_reg=0.0000 Loss=0.0088 grad_norm=0.0120 wnorm=49.9403 Time=0.0331
INFO     2024-05-01 14:36:14,056 Epoch time: 36.1014, Total time: 503.4619
INFO     2024-05-01 14:36:14,056 Test without loss fusion
INFO     2024-05-01 14:36:16,525 [14]: eps=0.00229930 active=0.3726 inactive=0.6260 CE=0.0079 Rob_Loss=0.0087 Rob_Err=0.0155 L_tightness=0.0000 L_relu=0.0000 L_std=0.1875 loss_reg=0.0000 Loss=0.0087 wnorm=49.9797 Time=0.0112
INFO     2024-05-01 14:36:16,575 
INFO     2024-05-01 14:36:16,575 Epoch 15, learning rate [0.002], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:36:52,486 [15]: eps=0.00261929 active=0.3710 inactive=0.6275 CE=0.0078 Rob_Loss=0.0084 Rob_Err=0.0152 L_tightness=0.0000 L_relu=0.0000 L_std=0.1919 loss_reg=0.0000 Loss=0.0084 grad_norm=0.0125 wnorm=50.0245 Time=0.0329
INFO     2024-05-01 14:36:52,487 Epoch time: 35.9128, Total time: 539.3747
INFO     2024-05-01 14:36:52,488 Test without loss fusion
INFO     2024-05-01 14:36:54,957 [15]: eps=0.00261929 active=0.3720 inactive=0.6263 CE=0.0077 Rob_Loss=0.0085 Rob_Err=0.0152 L_tightness=0.0000 L_relu=0.0000 L_std=0.1919 loss_reg=0.0000 Loss=0.0085 wnorm=50.0752 Time=0.0111
INFO     2024-05-01 14:36:55,004 
INFO     2024-05-01 14:36:55,004 Epoch 16, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:37:30,875 [16]: eps=0.00293928 active=0.3695 inactive=0.6288 CE=0.0074 Rob_Loss=0.0080 Rob_Err=0.0145 L_tightness=0.0000 L_relu=0.0000 L_std=0.1948 loss_reg=0.0000 Loss=0.0080 grad_norm=0.0102 wnorm=50.0813 Time=0.0329
INFO     2024-05-01 14:37:30,877 Epoch time: 35.8732, Total time: 575.2479
INFO     2024-05-01 14:37:30,877 Test without loss fusion
INFO     2024-05-01 14:37:33,346 [16]: eps=0.00293928 active=0.3689 inactive=0.6293 CE=0.0073 Rob_Loss=0.0080 Rob_Err=0.0144 L_tightness=0.0000 L_relu=0.0000 L_std=0.1928 loss_reg=0.0000 Loss=0.0080 wnorm=50.0864 Time=0.0112
INFO     2024-05-01 14:37:33,391 
INFO     2024-05-01 14:37:33,391 Epoch 17, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:38:09,296 [17]: eps=0.00325927 active=0.3685 inactive=0.6296 CE=0.0074 Rob_Loss=0.0080 Rob_Err=0.0145 L_tightness=0.0000 L_relu=0.0000 L_std=0.1958 loss_reg=0.0000 Loss=0.0080 grad_norm=0.0110 wnorm=50.0905 Time=0.0329
INFO     2024-05-01 14:38:09,298 Epoch time: 35.9068, Total time: 611.1547
INFO     2024-05-01 14:38:09,298 Test without loss fusion
INFO     2024-05-01 14:38:11,771 [17]: eps=0.00325927 active=0.3681 inactive=0.6300 CE=0.0074 Rob_Loss=0.0080 Rob_Err=0.0144 L_tightness=0.0000 L_relu=0.0000 L_std=0.1938 loss_reg=0.0000 Loss=0.0080 wnorm=50.0954 Time=0.0112
INFO     2024-05-01 14:38:11,817 
INFO     2024-05-01 14:38:11,817 Epoch 18, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:38:47,821 [18]: eps=0.00357926 active=0.3677 inactive=0.6303 CE=0.0074 Rob_Loss=0.0081 Rob_Err=0.0145 L_tightness=0.0000 L_relu=0.0000 L_std=0.1968 loss_reg=0.0000 Loss=0.0081 grad_norm=0.0115 wnorm=50.0990 Time=0.0331
INFO     2024-05-01 14:38:47,823 Epoch time: 36.0056, Total time: 647.1602
INFO     2024-05-01 14:38:47,823 Test without loss fusion
INFO     2024-05-01 14:38:50,288 [18]: eps=0.00357926 active=0.3674 inactive=0.6305 CE=0.0074 Rob_Loss=0.0080 Rob_Err=0.0144 L_tightness=0.0000 L_relu=0.0000 L_std=0.1948 loss_reg=0.0000 Loss=0.0080 wnorm=50.1035 Time=0.0111
INFO     2024-05-01 14:38:50,334 
INFO     2024-05-01 14:38:50,334 Epoch 19, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:39:26,122 [19]: eps=0.00389924 active=0.3670 inactive=0.6308 CE=0.0074 Rob_Loss=0.0081 Rob_Err=0.0145 L_tightness=0.0000 L_relu=0.0000 L_std=0.1978 loss_reg=0.0000 Loss=0.0081 grad_norm=0.0119 wnorm=50.1066 Time=0.0327
INFO     2024-05-01 14:39:26,124 Epoch time: 35.7901, Total time: 682.9504
INFO     2024-05-01 14:39:26,124 Test without loss fusion
INFO     2024-05-01 14:39:28,594 [19]: eps=0.00389924 active=0.3667 inactive=0.6310 CE=0.0074 Rob_Loss=0.0081 Rob_Err=0.0145 L_tightness=0.0000 L_relu=0.0000 L_std=0.1962 loss_reg=0.0000 Loss=0.0081 wnorm=50.1105 Time=0.0111
INFO     2024-05-01 14:39:28,640 
INFO     2024-05-01 14:39:28,640 Epoch 20, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:40:03,995 [20]: eps=0.00390000 active=0.3664 inactive=0.6313 CE=0.0074 Rob_Loss=0.0081 Rob_Err=0.0145 Loss=0.0081 grad_norm=0.0120 wnorm=50.1137 Time=0.0301
INFO     2024-05-01 14:40:03,997 Epoch time: 35.3563, Total time: 718.3067
INFO     2024-05-01 14:40:03,997 Test without loss fusion
INFO     2024-05-01 14:40:06,392 [20]: eps=0.00390000 active=0.3663 inactive=0.6315 CE=0.0073 Rob_Loss=0.0080 Rob_Err=0.0144 Loss=0.0080 wnorm=50.1181 Time=0.0092
INFO     2024-05-01 14:40:06,393 Best epoch 20, error 0.0000, robust error 0.0144
INFO     2024-05-01 14:40:06,451 
INFO     2024-05-01 14:40:06,451 Epoch 21, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:40:41,663 [21]: eps=0.00390000 active=0.3660 inactive=0.6318 CE=0.0074 Rob_Loss=0.0080 Rob_Err=0.0144 Loss=0.0080 grad_norm=0.0120 wnorm=50.1215 Time=0.0301
INFO     2024-05-01 14:40:41,665 Epoch time: 35.2136, Total time: 753.5203
INFO     2024-05-01 14:40:41,665 Test without loss fusion
INFO     2024-05-01 14:40:44,059 [21]: eps=0.00390000 active=0.3658 inactive=0.6320 CE=0.0073 Rob_Loss=0.0079 Rob_Err=0.0143 Loss=0.0079 wnorm=50.1260 Time=0.0092
INFO     2024-05-01 14:40:44,059 Best epoch 21, error 0.0000, robust error 0.0143
INFO     2024-05-01 14:40:44,148 
INFO     2024-05-01 14:40:44,148 Epoch 22, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:41:27,160 [22]: eps=0.00390000 active=0.3655 inactive=0.6322 CE=0.0073 Rob_Loss=0.0080 Rob_Err=0.0143 Loss=0.0080 grad_norm=0.0119 wnorm=50.1294 Time=0.0336
INFO     2024-05-01 14:41:27,163 Epoch time: 43.0149, Total time: 796.5353
INFO     2024-05-01 14:41:27,163 Test without loss fusion
INFO     2024-05-01 14:41:29,924 [22]: eps=0.00390000 active=0.3653 inactive=0.6325 CE=0.0072 Rob_Loss=0.0079 Rob_Err=0.0142 Loss=0.0079 wnorm=50.1343 Time=0.0105
INFO     2024-05-01 14:41:29,924 Best epoch 22, error 0.0000, robust error 0.0142
INFO     2024-05-01 14:41:30,013 
INFO     2024-05-01 14:41:30,013 Epoch 23, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:42:29,971 [23]: eps=0.00390000 active=0.3651 inactive=0.6327 CE=0.0073 Rob_Loss=0.0079 Rob_Err=0.0142 Loss=0.0079 grad_norm=0.0118 wnorm=50.1376 Time=0.0413
INFO     2024-05-01 14:42:29,974 Epoch time: 59.9603, Total time: 856.4956
INFO     2024-05-01 14:42:29,974 Test without loss fusion
INFO     2024-05-01 14:42:33,050 [23]: eps=0.00390000 active=0.3649 inactive=0.6329 CE=0.0072 Rob_Loss=0.0078 Rob_Err=0.0142 Loss=0.0078 wnorm=50.1426 Time=0.0116
INFO     2024-05-01 14:42:33,050 Best epoch 23, error 0.0000, robust error 0.0142
INFO     2024-05-01 14:42:33,139 
INFO     2024-05-01 14:42:33,139 Epoch 24, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:43:23,065 [24]: eps=0.00390000 active=0.3646 inactive=0.6333 CE=0.0072 Rob_Loss=0.0079 Rob_Err=0.0142 Loss=0.0079 grad_norm=0.0116 wnorm=50.1467 Time=0.0367
INFO     2024-05-01 14:43:23,068 Epoch time: 49.9290, Total time: 906.4246
INFO     2024-05-01 14:43:23,068 Test without loss fusion
INFO     2024-05-01 14:43:25,896 [24]: eps=0.00390000 active=0.3643 inactive=0.6336 CE=0.0072 Rob_Loss=0.0078 Rob_Err=0.0141 Loss=0.0078 wnorm=50.1531 Time=0.0107
INFO     2024-05-01 14:43:25,897 Best epoch 24, error 0.0000, robust error 0.0141
INFO     2024-05-01 14:43:25,986 
INFO     2024-05-01 14:43:25,986 Epoch 25, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:44:05,486 [25]: eps=0.00390000 active=0.3640 inactive=0.6339 CE=0.0072 Rob_Loss=0.0078 Rob_Err=0.0141 Loss=0.0078 grad_norm=0.0115 wnorm=50.1595 Time=0.0304
INFO     2024-05-01 14:44:05,488 Epoch time: 39.5011, Total time: 945.9257
INFO     2024-05-01 14:44:05,488 Test without loss fusion
INFO     2024-05-01 14:44:07,904 [25]: eps=0.00390000 active=0.3637 inactive=0.6342 CE=0.0071 Rob_Loss=0.0077 Rob_Err=0.0140 Loss=0.0077 wnorm=50.1691 Time=0.0093
INFO     2024-05-01 14:44:07,905 Best epoch 25, error 0.0000, robust error 0.0140
INFO     2024-05-01 14:44:08,003 
INFO     2024-05-01 14:44:08,003 Epoch 26, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:44:47,025 [26]: eps=0.00390000 active=0.3634 inactive=0.6344 CE=0.0072 Rob_Loss=0.0078 Rob_Err=0.0140 Loss=0.0078 grad_norm=0.0113 wnorm=50.1800 Time=0.0301
INFO     2024-05-01 14:44:47,026 Epoch time: 39.0231, Total time: 984.9488
INFO     2024-05-01 14:44:47,027 Test without loss fusion
INFO     2024-05-01 14:44:49,423 [26]: eps=0.00390000 active=0.3632 inactive=0.6347 CE=0.0071 Rob_Loss=0.0077 Rob_Err=0.0139 Loss=0.0077 wnorm=50.1962 Time=0.0092
INFO     2024-05-01 14:44:49,423 Best epoch 26, error 0.0000, robust error 0.0139
INFO     2024-05-01 14:44:49,537 
INFO     2024-05-01 14:44:49,537 Epoch 27, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:45:32,287 [27]: eps=0.00390000 active=0.3629 inactive=0.6350 CE=0.0071 Rob_Loss=0.0077 Rob_Err=0.0139 Loss=0.0077 grad_norm=0.0111 wnorm=50.2116 Time=0.0332
INFO     2024-05-01 14:45:32,288 Epoch time: 42.7514, Total time: 1027.7002
INFO     2024-05-01 14:45:32,289 Test without loss fusion
INFO     2024-05-01 14:45:34,685 [27]: eps=0.00390000 active=0.3626 inactive=0.6353 CE=0.0070 Rob_Loss=0.0076 Rob_Err=0.0137 Loss=0.0076 wnorm=50.2254 Time=0.0092
INFO     2024-05-01 14:45:34,685 Best epoch 27, error 0.0000, robust error 0.0137
INFO     2024-05-01 14:45:34,776 
INFO     2024-05-01 14:45:34,776 Epoch 28, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:46:09,102 [28]: eps=0.00390000 active=0.3624 inactive=0.6355 CE=0.0070 Rob_Loss=0.0076 Rob_Err=0.0137 Loss=0.0076 grad_norm=0.0108 wnorm=50.2416 Time=0.0290
INFO     2024-05-01 14:46:09,104 Epoch time: 34.3280, Total time: 1062.0281
INFO     2024-05-01 14:46:09,104 Test without loss fusion
INFO     2024-05-01 14:46:11,494 [28]: eps=0.00390000 active=0.3623 inactive=0.6357 CE=0.0069 Rob_Loss=0.0075 Rob_Err=0.0135 Loss=0.0075 wnorm=50.2610 Time=0.0092
INFO     2024-05-01 14:46:11,494 Best epoch 28, error 0.0000, robust error 0.0135
INFO     2024-05-01 14:46:11,592 
INFO     2024-05-01 14:46:11,592 Epoch 29, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:46:45,835 [29]: eps=0.00390000 active=0.3623 inactive=0.6356 CE=0.0069 Rob_Loss=0.0075 Rob_Err=0.0134 Loss=0.0075 grad_norm=0.0106 wnorm=50.2861 Time=0.0290
INFO     2024-05-01 14:46:45,836 Epoch time: 34.2439, Total time: 1096.2720
INFO     2024-05-01 14:46:45,836 Test without loss fusion
INFO     2024-05-01 14:46:48,226 [29]: eps=0.00390000 active=0.3624 inactive=0.6355 CE=0.0068 Rob_Loss=0.0074 Rob_Err=0.0132 Loss=0.0074 wnorm=50.3155 Time=0.0092
INFO     2024-05-01 14:46:48,226 Best epoch 29, error 0.0000, robust error 0.0132
INFO     2024-05-01 14:46:48,316 
INFO     2024-05-01 14:46:48,316 Epoch 30, learning rate [0.0004], dir model_coco_resv5_fast_ci_huber_5
INFO     2024-05-01 14:47:22,541 [30]: eps=0.00390000 active=0.3625 inactive=0.6354 CE=0.0067 Rob_Loss=0.0074 Rob_Err=0.0130 Loss=0.0074 grad_norm=0.0106 wnorm=50.3438 Time=0.0290
INFO     2024-05-01 14:47:22,543 Epoch time: 34.2267, Total time: 1130.4987
INFO     2024-05-01 14:47:22,543 Test without loss fusion
INFO     2024-05-01 14:47:24,933 [30]: eps=0.00390000 active=0.3626 inactive=0.6352 CE=0.0065 Rob_Loss=0.0072 Rob_Err=0.0127 Loss=0.0072 wnorm=50.3777 Time=0.0092
INFO     2024-05-01 14:47:24,933 Best epoch 30, error 0.0000, robust error 0.0127
INFO     2024-05-01 14:47:25,023 
'''

log_data_vanilla = '''

'''



def get_line(data):
    max_epoch = 30
    epochs = list(range(1, max_epoch + 1))
    ce_values = [None] * max_epoch
    rob_loss_values = [None] * max_epoch
    processed_epochs = set()
    for line in data.split('\n'):
        # Modified regular expression where 'Rob_Err' and 'Loss' are optional
        pattern = r'(\d+)\]:.*CE=(\d+\.\d+)(?:.*Rob_Loss=(\d+\.\d+))?' ##only Rob_Loss
        # pattern = r'(\d+)\]:.*CE=(\d+\.\d+)(?:.*Loss=(\d+\.\d+))?'
        match = re.search(pattern, line)

        if match:
            epoch = int(match.group(1))
            if epoch not in processed_epochs:
                ce = float(match.group(2))  # CE is always expected to be present
                rob_loss = float(match.group(3)) if match.group(3) else None
                ce_values[epoch - 1] = ce
                rob_loss_values[epoch - 1] = rob_loss
                processed_epochs.add(epoch)

    return ce_values, rob_loss_values, epochs


ce_values, rob_loss_values, epochs = get_line(log_data_fast)
ce_values_v, rob_loss_values_v, _ = get_line(log_data_vanilla)

# rob_loss_values_v[0] = 3
rob_loss_values[0] = 2.1


plt.figure(figsize=(8, 5))
plt.plot(epochs, ce_values, label='Regular Loss (Fast)', marker='o')
plt.plot(epochs, rob_loss_values, label='Robust Loss (Fast)', marker='x')

plt.plot(epochs, ce_values_v, label='Regular Loss (Vanilla)', marker='o', linestyle=':')
plt.plot(epochs, rob_loss_values_v, label='Robust Loss (Vanilla)', marker='x', linestyle=':')
plt.xlabel('Epoch')

# plt.ylabel('Value')

plt.ylabel('Value (log scale)')
plt.yscale('log')  # Set the y-axis to logarithmic scale

plt.legend()
plt.tight_layout()
plt.savefig('resnet9_fv_cibp_30_huber_5.png')
plt.show()



# Plotting
# plt.figure(figsize=(12, 6))
# plt.plot(epochs, ce_values, label='Regular MSE Loss', marker='o')
# plt.plot(epochs, rob_loss_values, label='Robustness Loss', marker='x')
# plt.xlabel('Epoch')
# plt.ylabel('Loss Value')
# plt.title('Training Metrics')

