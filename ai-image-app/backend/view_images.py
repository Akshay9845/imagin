#!/usr/bin/env python3
"""
View generated images and their information
"""

import os
import glob
from datetime import datetime

def list_generated_images():
    """List all generated images in the static directory"""
    static_dir = "static"
    
    if not os.path.exists(static_dir):
        print("❌ Static directory not found!")
        return
    
    # Get all PNG files
    image_files = glob.glob(os.path.join(static_dir, "*.png"))
    
    if not image_files:
        print("❌ No images found in static directory!")
        return
    
    print("🎨 Generated Images")
    print("=" * 50)
    print(f"📁 Directory: {os.path.abspath(static_dir)}")
    print(f"📊 Total images: {len(image_files)}")
    print()
    
    # Sort by modification time (newest first)
    image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        size_mb = os.path.getsize(image_path) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(image_path))
        
        print(f"{i:2d}. 📸 {filename}")
        print(f"    📏 Size: {size_mb:.1f} MB")
        print(f"    🕒 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    🌐 URL: http://localhost:5001/static/{filename}")
        print()
    
    print("💡 Tips:")
    print("   • Open the URLs in your browser to view images")
    print("   • Use the web interface at http://localhost:3000")
    print("   • Generate more images with: python generate_test_images.py")

def main():
    """Main function"""
    list_generated_images()

if __name__ == "__main__":
    main() 