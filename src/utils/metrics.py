"""
Resource monitoring and profiling utilities for Jetson Nano
"""
import psutil
import time
import json
from datetime import datetime
from pathlib import Path

class JetsonProfiler:
    """Monitor CPU, GPU, memory, and thermal state"""
    
    def _init_(self):
        self.results = {}
    
    def monitor_jetson_4gb(self):
        """Print current resource usage"""
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=1)
        mem_mb = process.memory_info().rss / 1024 / 1024
        jetson_nano_4gb_max_mem = 4096
        
        print(f"CPU: {cpu_percent:.1f}% | Mem: {mem_mb:.1f}MB / {jetson_nano_4gb_max_mem}MB", end="")
        
        if mem_mb > jetson_nano_4gb_max_mem:
            print(" | ⚠️  EXCEEDS 4GB LIMIT!")
        elif mem_mb > jetson_nano_4gb_max_mem * 0.85:
            print(" | ⚠️  WARNING: 85% memory threshold!")
        else:
            print()
        
        return {
            'cpu': cpu_percent,
            'memory': mem_mb,
            'timestamp': datetime.now().isoformat()
        }
    
    def log_results(self, phase, metrics, description):
        """Log metrics to file"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"{phase}{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        
        log_data = {
            'phase': phase,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n✓ Results logged to: {log_file}")

# Global profiler instance
profiler = JetsonProfiler()
