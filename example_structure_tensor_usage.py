#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于结构张量的SSIM使用示例

本脚本展示了如何在实际项目中使用基于结构张量的SSIM方法，
包括图像质量评估、算法效果比较等应用场景。
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

from ssim_structure_tensor import ssim_structure_tensor, SSIM_StructureTensor


def load_image_pair(img1_path, img2_path):
    """加载图像对"""
    try:
        img1 = Image.open(img1_path).convert('L')  # 转换为灰度图
        img2 = Image.open(img2_path).convert('L')
        return img1, img2
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None, None


def evaluate_image_quality(original_path, processed_path):
    """评估图像处理质量"""
    print(f"\n=== 图像质量评估 ===")
    print(f"原始图像: {original_path}")
    print(f"处理图像: {processed_path}")
    
    # 加载图像
    img1, img2 = load_image_pair(original_path, processed_path)
    if img1 is None or img2 is None:
        return None
    
    # 计算结构张量SSIM
    try:
        score = ssim_structure_tensor(img1, img2)
        print(f"结构张量SSIM: {score:.4f}")
        
        # 质量评级
        if score > 0.9:
            quality = "优秀"
        elif score > 0.7:
            quality = "良好"
        elif score > 0.5:
            quality = "一般"
        else:
            quality = "较差"
        
        print(f"质量评级: {quality}")
        return score
        
    except Exception as e:
        print(f"SSIM计算失败: {e}")
        return None


def batch_evaluate_directory(input_dir, output_dir):
    """批量评估目录中的图像"""
    print(f"\n=== 批量图像质量评估 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists() or not output_path.exists():
        print("目录不存在")
        return
    
    # 获取图像文件
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    input_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    results = []
    
    for input_file in input_files:
        output_file = output_path / input_file.name
        if output_file.exists():
            score = evaluate_image_quality(str(input_file), str(output_file))
            if score is not None:
                results.append((input_file.name, score))
    
    if results:
        print(f"\n=== 批量评估结果汇总 ===")
        print(f"{'文件名':<20} {'SSIM分数':<10} {'质量评级':<10}")
        print("-" * 45)
        
        total_score = 0
        for filename, score in results:
            if score > 0.9:
                quality = "优秀"
            elif score > 0.7:
                quality = "良好"
            elif score > 0.5:
                quality = "一般"
            else:
                quality = "较差"
            
            print(f"{filename:<20} {score:<10.4f} {quality:<10}")
            total_score += score
        
        avg_score = total_score / len(results)
        print("-" * 45)
        print(f"平均SSIM: {avg_score:.4f}")
        print(f"处理文件数: {len(results)}")
    
    return results


def compare_algorithms(original_img, algo1_img, algo2_img, 
                      algo1_name="算法1", algo2_name="算法2"):
    """比较两种算法的效果"""
    print(f"\n=== 算法效果比较 ===")
    
    # 计算SSIM分数
    score1 = ssim_structure_tensor(original_img, algo1_img)
    score2 = ssim_structure_tensor(original_img, algo2_img)
    
    print(f"{algo1_name} SSIM: {score1:.4f}")
    print(f"{algo2_name} SSIM: {score2:.4f}")
    
    # 比较结果
    if score1 > score2:
        winner = algo1_name
        diff = score1 - score2
    elif score2 > score1:
        winner = algo2_name
        diff = score2 - score1
    else:
        winner = "平局"
        diff = 0
    
    print(f"\n优胜算法: {winner}")
    if diff > 0:
        print(f"优势幅度: {diff:.4f}")
    
    return score1, score2


def create_quality_report(results, output_file="quality_report.txt"):
    """生成质量评估报告"""
    if not results:
        print("没有结果可生成报告")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("图像质量评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("基于结构张量的SSIM评估结果\n")
        f.write("-" * 30 + "\n\n")
        
        # 统计信息
        scores = [score for _, score in results]
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        std_score = np.std(scores)
        
        f.write(f"总文件数: {len(results)}\n")
        f.write(f"平均SSIM: {avg_score:.4f}\n")
        f.write(f"最高SSIM: {max_score:.4f}\n")
        f.write(f"最低SSIM: {min_score:.4f}\n")
        f.write(f"标准差: {std_score:.4f}\n\n")
        
        # 质量分布
        excellent = sum(1 for _, score in results if score > 0.9)
        good = sum(1 for _, score in results if 0.7 < score <= 0.9)
        fair = sum(1 for _, score in results if 0.5 < score <= 0.7)
        poor = sum(1 for _, score in results if score <= 0.5)
        
        f.write("质量分布:\n")
        f.write(f"优秀 (>0.9): {excellent} ({excellent/len(results)*100:.1f}%)\n")
        f.write(f"良好 (0.7-0.9): {good} ({good/len(results)*100:.1f}%)\n")
        f.write(f"一般 (0.5-0.7): {fair} ({fair/len(results)*100:.1f}%)\n")
        f.write(f"较差 (≤0.5): {poor} ({poor/len(results)*100:.1f}%)\n\n")
        
        # 详细结果
        f.write("详细结果:\n")
        f.write(f"{'文件名':<30} {'SSIM分数':<10} {'质量评级':<10}\n")
        f.write("-" * 55 + "\n")
        
        for filename, score in sorted(results, key=lambda x: x[1], reverse=True):
            if score > 0.9:
                quality = "优秀"
            elif score > 0.7:
                quality = "良好"
            elif score > 0.5:
                quality = "一般"
            else:
                quality = "较差"
            
            f.write(f"{filename:<30} {score:<10.4f} {quality:<10}\n")
    
    print(f"\n质量评估报告已保存到: {output_file}")


def demo_with_test_images():
    """使用测试图像进行演示"""
    print("=== 结构张量SSIM演示 ===")
    
    # 创建测试图像
    size = 256
    x, y = np.meshgrid(np.linspace(0, 10, size), np.linspace(0, 10, size))
    
    # 原始图像
    original = np.sin(x) * np.cos(y) + 0.5 * np.sin(2*x) * np.cos(3*y)
    original = (original - original.min()) / (original.max() - original.min())
    original_pil = Image.fromarray((original * 255).astype(np.uint8))
    
    # 创建不同处理版本
    # 1. 添加噪声
    noisy = original + 0.05 * np.random.randn(size, size)
    noisy = np.clip(noisy, 0, 1)
    noisy_pil = Image.fromarray((noisy * 255).astype(np.uint8))
    
    # 2. 模糊处理
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(original, sigma=1.0)
    blurred_pil = Image.fromarray((blurred * 255).astype(np.uint8))
    
    # 3. 边缘增强
    from scipy.ndimage import sobel
    edges = np.sqrt(sobel(original, axis=0)**2 + sobel(original, axis=1)**2)
    enhanced = original + 0.2 * edges
    enhanced = np.clip(enhanced, 0, 1)
    enhanced_pil = Image.fromarray((enhanced * 255).astype(np.uint8))
    
    # 评估不同处理的质量
    print("\n不同处理方法的质量评估:")
    
    algorithms = [
        ("噪声处理", noisy_pil),
        ("模糊处理", blurred_pil),
        ("边缘增强", enhanced_pil)
    ]
    
    results = []
    for name, img in algorithms:
        score = ssim_structure_tensor(original_pil, img)
        results.append((name, score))
        print(f"{name}: {score:.4f}")
    
    # 找出最佳算法
    best_algo, best_score = max(results, key=lambda x: x[1])
    print(f"\n最佳处理方法: {best_algo} (SSIM: {best_score:.4f})")
    
    return results


def main():
    """主函数 - 演示各种使用场景"""
    print("基于结构张量的SSIM使用示例")
    print("=" * 40)
    
    # 1. 基本演示
    demo_results = demo_with_test_images()
    
    # 2. 检查是否有实际图像可以测试
    test_input_dir = "detail_enhance/input"
    test_output_dir = "detail_enhance/output"
    
    if os.path.exists(test_input_dir) and os.path.exists(test_output_dir):
        print(f"\n发现测试目录，进行批量评估...")
        batch_results = batch_evaluate_directory(test_input_dir, test_output_dir)
        
        if batch_results:
            create_quality_report(batch_results, "structure_tensor_ssim_report.txt")
    else:
        print(f"\n未找到测试目录 {test_input_dir} 或 {test_output_dir}")
        print("跳过批量评估")
    
    # 3. 模块化使用示例
    print(f"\n=== 模块化使用示例 ===")
    
    # 创建SSIM模块
    ssim_module = SSIM_StructureTensor(channel=1, win_size=11)
    
    # 创建测试张量
    img1_tensor = torch.randn(1, 1, 128, 128)
    img2_tensor = img1_tensor + 0.1 * torch.randn(1, 1, 128, 128)
    
    # 计算相似性
    try:
        # 注意：这里需要转换为PIL图像格式
        img1_np = img1_tensor.squeeze().numpy()
        img2_np = img2_tensor.squeeze().numpy()
        
        # 归一化到0-1范围
        img1_np = (img1_np - img1_np.min()) / (img1_np.max() - img1_np.min())
        img2_np = (img2_np - img2_np.min()) / (img2_np.max() - img2_np.min())
        
        img1_pil = Image.fromarray((img1_np * 255).astype(np.uint8))
        img2_pil = Image.fromarray((img2_np * 255).astype(np.uint8))
        
        tensor_score = ssim_structure_tensor(img1_pil, img2_pil)
        print(f"张量SSIM分数: {tensor_score:.4f}")
        
    except Exception as e:
        print(f"张量计算失败: {e}")
    
    print(f"\n=== 使用建议 ===")
    print("1. 对于图像质量评估，建议SSIM > 0.7")
    print("2. 对于算法比较，关注SSIM差异 > 0.05")
    print("3. 对于实时应用，考虑降低窗口大小以提高速度")
    print("4. 对于高精度应用，可以增大窗口大小")
    
    print(f"\n演示完成！")


if __name__ == "__main__":
    main()