#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DENet特征提取器测试脚本
基于END.py的特征提取方法，专门用于DENet模型
"""

import torch
import argparse
import os
from denet_feature_extractor import DENetFeatureExtractor

def test_denet_feature_extraction(weights_path, image_size=256, batch_size=1, save_features=True):
    """
    测试DENet特征提取功能
    
    参数:
    weights_path (str): 预训练DENet模型路径
    image_size (int): 输入图像尺寸
    batch_size (int): 批次大小
    save_features (bool): 是否保存特征
    """
    print("=== DENet特征提取测试 ===")
    print(f"模型路径: {weights_path}")
    print(f"图像尺寸: {image_size}x{image_size}")
    print(f"批次大小: {batch_size}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 初始化特征提取器
        print("\n初始化DENet特征提取器...")
        extractor = DENetFeatureExtractor(weights_path, base_dim=16)
        
        # 创建测试输入
        print(f"\n创建测试输入: [{batch_size}, 1, {image_size}, {image_size}]")
        test_input = torch.randn(batch_size, 1, image_size, image_size).to(device)
        
        # 提取特征
        print("\n开始特征提取...")
        save_dir = 'denet_test_features' if save_features else None
        features = extractor.extract_features(test_input, save_dir=save_dir)
        
        # 显示结果
        print("\n=== 特征提取结果 ===")
        feature_names = ['f_d_vi', 'f_d_vis1', 'f_d_vis2', 'f_d_vis3']
        for name, feat in zip(feature_names, features):
            print(f"{name:10} | 形状: {str(feat.shape):25} | 数据类型: {feat.dtype} | 设备: {feat.device}")
            print(f"{'':10} | 最小值: {feat.min().item():.6f} | 最大值: {feat.max().item():.6f} | 平均值: {feat.mean().item():.6f}")
            print("-" * 80)
        
        # 测试中间层特征提取
        print("\n=== 中间层特征提取测试 ===")
        intermediate_features = extractor.extract_intermediate_features(test_input)
        for name, feat in intermediate_features.items():
            print(f"{name:15} | 形状: {str(feat.shape):25}")
        
        # 可视化特征（如果保存了特征）
        if save_features:
            print("\n生成特征可视化...")
            extractor.visualize_features(features, save_dir='denet_test_features')
        
        print("\n✅ 特征提取测试成功完成！")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 特征提取过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_direct_model_call(weights_path, image_size=256):
    """
    比较特征提取器和直接调用模型的结果
    """
    print("\n=== 与直接模型调用结果比较 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_input = torch.randn(1, 1, image_size, image_size).to(device)
    
    try:
        # 使用特征提取器
        extractor = DENetFeatureExtractor(weights_path, base_dim=16)
        extractor_features = extractor.extract_features(test_input)
        
        # 直接调用模型
        from detail_enhance.encoder0716 import DENet
        model = DENet(base_dim=16)
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            direct_features = model(test_input)
        
        # 比较结果
        print("特征形状比较:")
        feature_names = ['f_d_vi', 'f_d_vis1', 'f_d_vis2', 'f_d_vis3']
        for i, name in enumerate(feature_names):
            extractor_shape = extractor_features[i].shape
            direct_shape = direct_features[i].shape
            match = "✅" if extractor_shape == direct_shape else "❌"
            print(f"{name:10} | 提取器: {str(extractor_shape):20} | 直接调用: {str(direct_shape):20} | {match}")
        
        # 比较数值差异
        print("\n数值差异比较:")
        for i, name in enumerate(feature_names):
            diff = torch.abs(extractor_features[i] - direct_features[i]).max().item()
            print(f"{name:10} | 最大差异: {diff:.10f}")
        
        print("\n✅ 比较完成！")
        
    except Exception as e:
        print(f"❌ 比较过程中出现错误: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='DENet特征提取器测试')
    parser.add_argument('--weights', type=str, 
                        default='d:/WVPFusion/detail_enhance/DeEn_model_epoch_1000.pth',
                        help='预训练DENet模型路径')
    parser.add_argument('--image_size', type=int, default=256,
                        help='输入图像尺寸')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--no_save', action='store_true',
                        help='不保存特征文件')
    parser.add_argument('--compare', action='store_true',
                        help='与直接模型调用结果进行比较')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.weights):
        print(f"❌ 找不到模型文件: {args.weights}")
        print("请检查路径是否正确")
        return
    
    # 运行测试
    success = test_denet_feature_extraction(
        args.weights, 
        args.image_size, 
        args.batch_size, 
        not args.no_save
    )
    
    # 运行比较测试
    if success and args.compare:
        compare_with_direct_model_call(args.weights, args.image_size)
    
    print("\n测试完成！")

if __name__ == '__main__':
    main()