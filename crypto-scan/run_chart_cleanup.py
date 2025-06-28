#!/usr/bin/env python3
"""
Standalone Chart Cleanup Tool
Manual execution tool for cleaning up old chart screenshots
"""

import sys
import os
from pathlib import Path

# Add crypto-scan directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main execution for chart cleanup"""
    print("🧹 CRYPTO SCANNER - Chart Cleanup Tool")
    print("=" * 50)
    
    try:
        from utils.chart_cleanup import cleanup_with_size_report
        
        # Show current disk usage first
        print("📊 Analyzing current chart storage...")
        
        # Run cleanup with detailed report
        cleanup_stats = cleanup_with_size_report(max_age_hours=72, dry_run=False)
        
        print("\n" + "=" * 50)
        print("🎯 CLEANUP SUMMARY:")
        print(f"• Initial storage: {cleanup_stats.get('initial_size_gb', 0):.2f} GB")
        print(f"• Files deleted: {cleanup_stats.get('deleted_files', 0)}")
        print(f"• Files preserved: {cleanup_stats.get('preserved_files', 0)}")
        print(f"• Space saved: {cleanup_stats.get('space_saved_gb', 0):.2f} GB")
        print(f"• Remaining storage: {cleanup_stats.get('remaining_size_gb', 0):.2f} GB")
        print(f"• Errors: {cleanup_stats.get('errors', 0)}")
        
        if cleanup_stats.get('deleted_files', 0) > 0:
            print(f"\n✅ Successfully cleaned up {cleanup_stats['deleted_files']} old chart files")
            print(f"💾 Freed {cleanup_stats.get('space_saved_gb', 0):.2f} GB of disk space")
        else:
            print("\n📂 No files were eligible for cleanup")
            
        print("\n🔍 Cleanup criteria:")
        print("• Files older than 72 hours")
        print("• Only processed training charts")
        print("• Preserved unprocessed charts")
        print("• Preserved recent charts")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the crypto-scan directory")
        return 1
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)