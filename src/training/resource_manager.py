"""
Local resource manager for monitoring GPU usage, temperature, and power consumption.
"""

import time
import threading
from typing import Dict, List, Optional
from datetime import datetime

from loguru import logger

try:
    import GPUtil
    import psutil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    logger.warning("GPU monitoring not available. Install with: pip install GPUtil psutil")

try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    logger.warning("NVIDIA ML not available. Install with: pip install nvidia-ml-py3")


class LocalResourceManager:
    """Manages local GPU resources and monitoring."""
    
    def __init__(self, max_power_watts: float = 450.0, max_temp_celsius: float = 85.0):
        """Initialize resource manager."""
        self.max_power_watts = max_power_watts
        self.max_temp_celsius = max_temp_celsius
        self.monitoring = False
        self.monitor_thread = None
        self.current_stats = {}
        
        # Check available GPUs
        self.available_gpus = self._detect_gpus()
        logger.info(f"Detected {len(self.available_gpus)} GPUs")
        
    def _detect_gpus(self) -> List[Dict]:
        """Detect available GPUs."""
        gpus = []
        
        if GPU_MONITORING_AVAILABLE:
            try:
                gpu_list = GPUtil.getGPUs()
                for i, gpu in enumerate(gpu_list):
                    gpus.append({
                        "id": i,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_free": gpu.memoryFree,
                        "memory_used": gpu.memoryUsed,
                        "temperature": gpu.temperature,
                        "load": gpu.load
                    })
            except Exception as e:
                logger.warning(f"Failed to detect GPUs with GPUtil: {e}")
        
        return gpus
    
    def get_gpu_stats(self, gpu_id: int = 0) -> Dict:
        """Get current GPU statistics."""
        stats = {
            "gpu_id": gpu_id,
            "utilization": 0.0,
            "memory_used": 0.0,
            "memory_total": 0.0,
            "temperature": 0.0,
            "power_usage": 0.0,
            "available": False
        }
        
        if GPU_MONITORING_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpu_id < len(gpus):
                    gpu = gpus[gpu_id]
                    stats.update({
                        "utilization": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature,
                        "available": True
                    })
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")
        
        # Try to get power usage with nvidia-ml-py3
        if NVIDIA_ML_AVAILABLE:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                stats["power_usage"] = power_usage
            except Exception as e:
                logger.warning(f"Failed to get power usage: {e}")
        
        return stats
    
    def check_resources(self, gpu_id: int = 0) -> bool:
        """Check if resources are within safe limits."""
        stats = self.get_gpu_stats(gpu_id)
        
        if not stats["available"]:
            return False
        
        # Check temperature
        if stats["temperature"] > self.max_temp_celsius:
            logger.warning(f"GPU temperature too high: {stats['temperature']}°C > {self.max_temp_celsius}°C")
            return False
        
        # Check power usage
        if stats["power_usage"] > self.max_power_watts:
            logger.warning(f"GPU power usage too high: {stats['power_usage']}W > {self.max_power_watts}W")
            return False
        
        return True
    
    def get_available_memory(self, gpu_id: int = 0) -> float:
        """Get available GPU memory in GB."""
        stats = self.get_gpu_stats(gpu_id)
        if stats["available"]:
            return (stats["memory_total"] - stats["memory_used"]) / 1024.0  # Convert to GB
        return 0.0
    
    def can_fit_model(self, model_size_gb: float, gpu_id: int = 0) -> bool:
        """Check if model can fit in GPU memory."""
        available_memory = self.get_available_memory(gpu_id)
        # Add 20% buffer for training overhead
        required_memory = model_size_gb * 1.2
        
        can_fit = available_memory >= required_memory
        logger.info(f"Model size: {model_size_gb}GB, Available: {available_memory}GB, Can fit: {can_fit}")
        
        return can_fit
    
    def start_monitoring(self, interval_seconds: int = 10) -> None:
        """Start continuous resource monitoring."""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self, interval_seconds: int) -> None:
        """Continuous monitoring loop."""
        while self.monitoring:
            try:
                for gpu in self.available_gpus:
                    gpu_id = gpu["id"]
                    stats = self.get_gpu_stats(gpu_id)
                    self.current_stats[gpu_id] = {
                        **stats,
                        "timestamp": datetime.now()
                    }
                    
                    # Check for thermal throttling
                    if stats["temperature"] > self.max_temp_celsius * 0.9:  # 90% of max
                        logger.warning(f"GPU {gpu_id} approaching thermal limit: {stats['temperature']}°C")
                    
                    # Check for high power usage
                    if stats["power_usage"] > self.max_power_watts * 0.9:  # 90% of max
                        logger.warning(f"GPU {gpu_id} approaching power limit: {stats['power_usage']}W")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def get_current_stats(self) -> Dict:
        """Get current monitoring statistics."""
        return self.current_stats.copy()
    
    def emergency_throttle(self, gpu_id: int = 0) -> None:
        """Emergency throttling for overheating/overpower."""
        logger.warning(f"Emergency throttling GPU {gpu_id}")
        
        # In a real implementation, this would:
        # 1. Reduce training batch size
        # 2. Lower GPU clock speeds
        # 3. Pause training temporarily
        # 4. Increase fan speeds
        
        # For now, just log the action
        stats = self.get_gpu_stats(gpu_id)
        logger.warning(f"GPU {gpu_id} stats: {stats}")
    
    def estimate_training_time(self, model_size: str, dataset_size: int) -> float:
        """Estimate training time in hours."""
        # Simple estimation based on model size and dataset
        base_times = {
            "7B": 3.0,   # hours for 1000 samples
            "13B": 6.0,
            "20B": 12.0
        }
        
        base_time = base_times.get(model_size, 3.0)
        scaling_factor = dataset_size / 1000.0
        
        estimated_time = base_time * scaling_factor
        logger.info(f"Estimated training time: {estimated_time:.1f} hours")
        
        return estimated_time
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "gpu_count": len(self.available_gpus),
            "gpus": self.available_gpus
        }
        
        return info