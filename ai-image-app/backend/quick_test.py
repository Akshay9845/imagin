#!/usr/bin/env python3
"""
Quick test for frontend-backend connection
"""

import requests

def quick_test():
    """Quick test of the connection"""
    print("🔍 Quick Connection Test")
    print("=" * 30)
    
    # Test backend
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        print("✅ Backend: Running on port 5001")
    except Exception as e:
        print(f"❌ Backend: {e}")
        return False
    
    # Test frontend
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        print("✅ Frontend: Running on port 3000")
    except Exception as e:
        print(f"❌ Frontend: {e}")
        return False
    
    print("\n🎉 Both services are running!")
    print("\n📝 You can now:")
    print("   1. Open http://localhost:3000 in your browser")
    print("   2. Enter 'forest' in the prompt field")
    print("   3. Click 'Generate' to create your image")
    print("\n⚠️  Note: First generation may take 1-2 minutes as the model loads")
    
    return True

if __name__ == "__main__":
    quick_test() 