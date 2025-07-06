"""
Robust Memory Handler - JSON Error Recovery System
Fixes memory/trader_outcomes.json loading errors with automatic corruption detection and recovery
"""

import json
import os
import time
import shutil
from typing import Dict, Any, Optional

class RobustMemoryHandler:
    """
    Handles memory file operations with automatic JSON corruption detection and recovery
    """
    
    def __init__(self):
        self.memory_files = [
            "memory/trader_outcomes.json",
            "data/tjdememory_outcomes.json", 
            "data/token_memory.json",
            "data/context/token_context_history.json"
        ]
    
    def load_memory_safe(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Safely load memory file with automatic corruption detection and recovery
        
        Args:
            file_path: Path to memory file
            
        Returns:
            Loaded data dictionary or None if critical error
        """
        try:
            # Handle empty or invalid file paths
            if not file_path or not file_path.strip():
                print(f"[MEMORY CRITICAL ERROR] {file_path}: Empty file path provided")
                return None
            
            # Ensure directory exists (handle case where dirname might be empty)
            dir_path = os.path.dirname(file_path)
            if dir_path:  # Only create directory if path has a directory component
                os.makedirs(dir_path, exist_ok=True)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"[MEMORY] {file_path} does not exist - creating fresh file")
                return self._create_fresh_memory_file(file_path)
            
            # Check if file is empty
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                print(f"[MEMORY] {file_path} is empty - reinitializing")
                return self._create_fresh_memory_file(file_path)
            
            # Try to parse JSON
            try:
                data = json.loads(content)
                print(f"✅ [MEMORY] {file_path} loaded successfully")
                return data
                
            except json.JSONDecodeError as je:
                print(f"🧨 [MEMORY ERROR] JSON corruption in {file_path}: {je}")
                print(f"    Error at line {je.lineno}, column {je.colno}: {je.msg}")
                
                # Backup corrupted file
                backup_path = f"{file_path}.corrupted.{int(time.time())}"
                shutil.copy2(file_path, backup_path)
                print(f"    Corrupted file backed up to: {backup_path}")
                
                # Create fresh file
                fresh_data = self._create_fresh_memory_file(file_path)
                print(f"    Created fresh memory file")
                return fresh_data
                
        except Exception as e:
            print(f"[MEMORY CRITICAL ERROR] {file_path}: {e}")
            return None
    
    def save_memory_safe(self, file_path: str, data: Dict[str, Any]) -> bool:
        """
        Safely save memory file with atomic write operations
        
        Args:
            file_path: Path to memory file
            data: Data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle empty or invalid file paths
            if not file_path or not file_path.strip():
                print(f"[MEMORY SAVE ERROR] Empty file path provided")
                return False
            
            # Ensure directory exists (handle case where dirname might be empty)
            dir_path = os.path.dirname(file_path)
            if dir_path:  # Only create directory if path has a directory component
                os.makedirs(dir_path, exist_ok=True)
            
            # Use atomic write (temp file + rename)
            temp_path = f"{file_path}.tmp.{int(time.time())}"
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            os.rename(temp_path, file_path)
            print(f"✅ [MEMORY] {file_path} saved successfully")
            return True
            
        except Exception as e:
            print(f"[MEMORY SAVE ERROR] {file_path}: {e}")
            # Clean up temp file if it exists
            temp_path = f"{file_path}.tmp.{int(time.time())}"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    
    def _create_fresh_memory_file(self, file_path: str) -> Dict[str, Any]:
        """
        Create fresh memory file with proper structure based on filename
        
        Args:
            file_path: Path to memory file
            
        Returns:
            Initial data structure
        """
        try:
            # Initialize with appropriate structure based on filename
            if "outcomes" in file_path.lower():
                initial_data = {"outcomes": []}
            elif "token_context_history" in file_path.lower():
                initial_data = {}
            elif "token_memory" in file_path.lower():
                initial_data = {}
            elif "trader_outcomes" in file_path.lower():
                initial_data = {"outcomes": [], "statistics": {"total": 0, "correct": 0, "accuracy": 0.0}}
            else:
                initial_data = {}
            
            # Save initial structure
            self.save_memory_safe(file_path, initial_data)
            return initial_data
            
        except Exception as e:
            print(f"[MEMORY CREATE ERROR] {file_path}: {e}")
            return {}
    
    def check_all_memory_files(self) -> Dict[str, bool]:
        """
        Check and repair all memory files in the system
        
        Returns:
            Status dictionary for each file
        """
        status = {}
        
        for file_path in self.memory_files:
            full_path = os.path.join("crypto-scan", file_path) if not file_path.startswith("crypto-scan") else file_path
            data = self.load_memory_safe(full_path)
            status[file_path] = data is not None
        
        return status
    
    def fix_memory_loading_errors(self) -> bool:
        """
        Comprehensive fix for all memory loading errors
        
        Returns:
            True if all files are working, False if any critical errors
        """
        print("🔧 [MEMORY FIX] Starting comprehensive memory file repair...")
        
        status = self.check_all_memory_files()
        all_good = True
        
        for file_path, is_working in status.items():
            if is_working:
                print(f"✅ {file_path}: Working correctly")
            else:
                print(f"❌ {file_path}: Failed to load/repair")
                all_good = False
        
        if all_good:
            print("🎉 [MEMORY FIX] All memory files operational!")
        else:
            print("⚠️  [MEMORY FIX] Some files still have issues")
        
        return all_good

# Global instance for easy access
memory_handler = RobustMemoryHandler()

def load_memory_safe(file_path: str) -> Optional[Dict[str, Any]]:
    """Convenience function for safe memory loading"""
    return memory_handler.load_memory_safe(file_path)

def save_memory_safe(file_path: str, data: Dict[str, Any]) -> bool:
    """Convenience function for safe memory saving"""
    return memory_handler.save_memory_safe(file_path, data)

def fix_all_memory_errors() -> bool:
    """Convenience function to fix all memory errors"""
    return memory_handler.fix_memory_loading_errors()