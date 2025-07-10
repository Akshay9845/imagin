#!/usr/bin/env python3
"""
Test script for memory-optimized video generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultra_realistic_video_system import UltraRealisticVideoSystem
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_optimized_generation():
    """Test the memory-optimized video generation"""
    
    logger.info("🧪 Testing memory-optimized video generation...")
    
    # Initialize the system
    video_system = UltraRealisticVideoSystem()
    
    # Test with very small parameters
    test_prompt = "A beautiful sunset"
    duration = 3  # Very short
    fps = 8       # Low FPS
    width = 256   # Small dimensions
    height = 256  # Small dimensions
    
    logger.info(f"Testing with parameters: {width}x{height}, {duration}s, {fps}fps")
    
    try:
        # Test direct text-to-video generation
        result = video_system.generate_direct_text_to_video(
            prompt=test_prompt,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            pipeline_type="modelscope_t2v"
        )
        
        logger.info(f"✅ Test successful! Video saved to: {result}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_memory_optimized_generation()
    if success:
        print("🎉 Memory optimization test passed!")
    else:
        print("💥 Memory optimization test failed!") 