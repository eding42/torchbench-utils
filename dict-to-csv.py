import csv

# Given dictionary
latencies = {'BERT_pytorch': 22.099, 'Background_Matting': -1, 'LearningToPaint': 8105.1045, 'Super_SloMo': 134.6809, 'alexnet': 4.2427, 'basic_gnn_edgecnn': 7.1637, 'basic_gnn_gcn': 16.1108, 'basic_gnn_gin': 0.9943, 'basic_gnn_sage': 2.0019, 'cm3leon_generate': 3676.9433, 'dcgan': 3.03, 'demucs': 222.5301, 'densenet121': 121.137, 'dlrm': 5.5073, 'doctr_det_predictor': 52.5351, 'doctr_reco_predictor': 12.0295, 'drq': 3.5215, 'fastNLP_Bert': 94.61, 'functorch_dp_cifar10': 2.1894, 'functorch_maml_omniglot': -1, 'hf_Albert': 17.7015, 'hf_Bart': 16.0214, 'hf_Bert': 19.164, 'hf_Bert_large': 41.8196, 'hf_BigBird': 461.266, 'hf_DistilBert': 12.7803, 'hf_GPT2': 20.5278, 'hf_GPT2_large': 130.0856, 'hf_Longformer': 178.3968, 'hf_Reformer': 31.5056, 'hf_Roberta_base': 31.0361, 'hf_T5': 31.7404, 'hf_T5_base': 74.2199, 'hf_T5_generate': 6260.2536, 'hf_T5_large': 101.6388, 'hf_Whisper': 9.6721, 'hf_clip': 31.8614, 'hf_distil_whisper': 47.3676, 'lennard_jones': 2.0526, 'llama': 9.1606, 'llama_v2_7b_16h': -1, 'maml': 1479.8509, 'maml_omniglot': -1, 'microbench_unbacked_tolist_sum': 356.0052, 'mnasnet1_0': 15.9439, 'mobilenet_v2': 9.2181, 'mobilenet_v2_quantized_qat': -1, 'mobilenet_v3_large': 12.4616, 'moco': -1, 'moondream': 225.4487, 'nanogpt': 2347.6628, 'nvidia_deeprecommender': 2.2111, 'opacus_cifar10': 3.8981, 'phlippe_densenet': 15.8093, 'phlippe_resnet': 2.5564, 'pyhpc_equation_of_state': 15.7579, 'pyhpc_isoneutral_mixing': 29.4034, 'pyhpc_turbulent_kinetic_energy': 21.9656, 'pytorch_CycleGAN_and_pix2pix': 8.0945, 'pytorch_stargan': 14.3405, 'pytorch_unet': 95.2666, 'resnet152': 67.8727, 'resnet18': 1.4086, 'resnet50': 27.8849, 'resnet50_quantized_qat': -1, 'resnext50_32x4d': 10.7623, 'sam': 932.4328, 'sam_fast': -1, 'shufflenet_v2_x1_0': 13.1738, 'simple_gpt': -1, 'simple_gpt_tp_manual': -1, 'soft_actor_critic': -1, 'speech_transformer': 16044.5095, 'squeezenet1_1': 1.1218, 'stable_diffusion_text_encoder': -1, 'stable_diffusion_unet': -1, 'tacotron2': 2302.3392, 'timm_efficientdet': -1, 'timm_efficientnet': 26.1868, 'timm_nfnet': 61.767, 'timm_regnet': 42.1139, 'timm_resnest': 12.1496, 'timm_vision_transformer': 17.7462, 'timm_vision_transformer_large': 213.8805, 'timm_vovnet': 30.7242, 'torch_multimodal_clip': 80.5256, 'tts_angular': 9.3958, 'vgg16': 7.9162, 'vision_maskrcnn': -1, 'yolov3': 105.5392}

# Convert to CSV format (tab-separated for easy pasting into Excel/Sheets)
csv_data = "Model Name\tLatency\n"
csv_data += "\n".join([f"{model}\t{latency}" for model, latency in latencies.items()])

# Save to a file (optional)
# with open("latencies.tsv", "w", encoding="utf-8") as f:
#     f.write(csv_data)

# Print to console for copying
print(csv_data)
