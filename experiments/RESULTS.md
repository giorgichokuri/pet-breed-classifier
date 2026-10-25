# Results

## Experiments (test set: 3,669 images, 37 breeds)

| run                         | model              | pretrained   | frozen   | aug   |   params (M) |   trainable (M) |   epochs |   best epoch |   val acc |   test acc |   top-5 |   macro F1 |   cat/dog acc |   train min |
|:----------------------------|:-------------------|:-------------|:---------|:------|-------------:|----------------:|---------:|-------------:|----------:|-----------:|--------:|-----------:|--------------:|------------:|
| e1_simple_cnn_scratch       | simple_cnn         | False        | False    | basic |         0.4  |            0.4  |       30 |           29 |    0.4293 |     0.3047 |  0.6787 |     0.2959 |        0.7855 |       11.19 |
| e2_resnet18_linear_probe    | resnet18           | True         | True     | basic |        11.2  |            0.02 |       10 |           10 |    0.9076 |     0.8883 |  0.9899 |     0.8863 |        0.9899 |        3.62 |
| e3_resnet18_finetune        | resnet18           | True         | False    | basic |        11.2  |           11.2  |       10 |            5 |    0.9312 |     0.8809 |  0.9836 |     0.8775 |        0.9913 |        3.78 |
| e4_resnet18_finetune_noaug  | resnet18           | True         | False    | none  |        11.2  |           11.2  |       13 |            8 |    0.9275 |     0.8937 |  0.9817 |     0.8909 |        0.9924 |        3.89 |
| e5_efficientnet_b0_finetune | efficientnet_b0    | True         | False    | basic |         4.05 |            4.05 |       15 |           10 |    0.9402 |     0.9131 |  0.9926 |     0.9117 |        0.9959 |        6.3  |
| e6_mobilenet_v3_finetune    | mobilenet_v3_large | True         | False    | basic |         4.25 |            4.25 |       10 |            5 |    0.9366 |     0.8962 |  0.9907 |     0.8951 |        0.9932 |        4.06 |

## Optimization: e5_efficientnet_b0_finetune (efficientnet_b0)

ONNX vs PyTorch max abs difference: 1.4e-04; evaluated on 3669 test images; CPU threads: 1

| variant                                 |   size_mb |   latency_ms |   test_acc |
|:----------------------------------------|----------:|-------------:|-----------:|
| PyTorch FP32                            |     16.5  |        38.63 |     0.9131 |
| ONNX FP32                               |     16.21 |        15.78 |     0.9131 |
| ONNX INT8 (conv_gemm_u8s8_reduce_range) |      4.84 |        29.38 |     0.4028 |

### INT8 quantization recipes (chosen by validation accuracy: conv_gemm_u8s8_reduce_range; FP32 validation accuracy 0.9402)

| recipe                      |   val_acc |   size_mb |   latency_ms |   quantize_s |
|:----------------------------|----------:|----------:|-------------:|-------------:|
| all_ops_u8s8                |    0.3478 |      4.93 |        18.56 |         34.5 |
| all_ops_u8s8_reduce_range   |    0.3478 |      4.93 |        18.65 |         33.8 |
| all_ops_u8u8                |    0.375  |      4.96 |        25.7  |         37.9 |
| conv_gemm_u8s8_reduce_range |    0.3822 |      4.84 |        28.61 |         36.9 |

### Most confused breed pairs (best model)

| true                       | predicted                  |   count |
|:---------------------------|:---------------------------|--------:|
| American Pit Bull Terrier  | Staffordshire Bull Terrier |      38 |
| Abyssinian                 | Bengal                     |      16 |
| Birman                     | Ragdoll                    |      15 |
| Ragdoll                    | Birman                     |      15 |
| Egyptian Mau               | Bengal                     |      12 |
| American Pit Bull Terrier  | American Bulldog           |      11 |
| Staffordshire Bull Terrier | American Bulldog           |       9 |
| Boxer                      | American Bulldog           |       6 |
| British Shorthair          | Russian Blue               |       6 |
| Russian Blue               | Bombay                     |       6 |
