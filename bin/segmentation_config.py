"""
Configuration file for segmentation performance optimization
"""
import multiprocessing
import os

class SegmentationConfig:
    """Configuration class for segmentation performance tuning"""
    
    # Parallel processing settings
    MAX_WORKERS = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    CHUNK_SIZE = 100  # Process tiles in chunks for memory management
    
    # I/O optimization
    COMPRESSION_LEVEL = 1  # TIFF compression level (0-9, lower = faster)
    BATCH_SIZE = 50  # Database batch operations size
    
    # Memory management
    MAX_MEMORY_GB = 16  # Maximum memory usage in GB
    CLEANUP_INTERVAL = 10  # Cleanup memory every N tiles
    
    # Progress reporting
    PROGRESS_INTERVAL = 5  # Report progress every N tiles
    
    # Database optimization
    USE_TRANSACTIONS = True  # Use database transactions for batch operations
    BATCH_TIMEOUT = 300  # Timeout for batch operations in seconds
    
    # Image processing
    SKIP_NORMALIZATION_IF_ALIGNED = True  # Skip normalization for aligned tiles
    USE_FAST_THRESHOLDING = True  # Use faster thresholding methods when possible
    
    @classmethod
    def get_optimal_workers(cls, available_memory_gb=None):
        """Get optimal number of workers based on available memory"""
        if available_memory_gb is None:
            available_memory_gb = cls.MAX_MEMORY_GB
        
        # Estimate memory per worker (rough estimate)
        memory_per_worker = 2  # GB per worker
        
        optimal_workers = min(
            multiprocessing.cpu_count(),
            int(available_memory_gb / memory_per_worker),
            cls.MAX_WORKERS
        )
        
        return max(1, optimal_workers)
    
    @classmethod
    def get_chunk_size(cls, total_tiles, available_memory_gb=None):
        """Get optimal chunk size for processing"""
        if available_memory_gb is None:
            available_memory_gb = cls.MAX_MEMORY_GB
        
        # Adjust chunk size based on available memory
        if available_memory_gb < 8:
            return min(25, total_tiles)
        elif available_memory_gb < 16:
            return min(50, total_tiles)
        else:
            return min(cls.CHUNK_SIZE, total_tiles)
    
    @classmethod
    def get_compression_settings(cls):
        """Get optimal compression settings for current system"""
        return [
            cv2.IMWRITE_TIFF_COMPRESSION, cls.COMPRESSION_LEVEL
        ]

# Performance monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.tile_times = []
        self.memory_usage = []
        self.db_operation_times = []
    
    def start_timing(self):
        """Start timing the overall process"""
        import time
        self.start_time = time.time()
    
    def record_tile_time(self, tile_id, processing_time):
        """Record time taken to process a tile"""
        self.tile_times.append((tile_id, processing_time))
    
    def record_db_operation(self, operation_type, duration):
        """Record database operation duration"""
        self.db_operation_times.append((operation_type, duration))
    
    def get_statistics(self):
        """Get performance statistics"""
        if not self.tile_times:
            return {}
        
        import numpy as np
        
        tile_durations = [t[1] for t in self.tile_times]
        
        stats = {
            'total_tiles': len(self.tile_times),
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'avg_tile_time': np.mean(tile_durations),
            'min_tile_time': np.min(tile_durations),
            'max_tile_time': np.max(tile_durations),
            'std_tile_time': np.std(tile_durations),
            'total_db_operations': len(self.db_operation_times)
        }
        
        if self.db_operation_times:
            db_durations = [d[1] for d in self.db_operation_times]
            stats['avg_db_time'] = np.mean(db_durations)
            stats['total_db_time'] = np.sum(db_durations)
        
        return stats
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total tiles processed: {stats['total_tiles']}")
        print(f"Total processing time: {stats['total_time']:.2f}s")
        print(f"Average time per tile: {stats['avg_tile_time']:.2f}s")
        print(f"Fastest tile: {stats['min_tile_time']:.2f}s")
        print(f"Slowest tile: {stats['max_tile_time']:.2f}s")
        print(f"Standard deviation: {stats['std_tile_time']:.2f}s")
        
        if 'total_db_operations' in stats:
            print(f"Database operations: {stats['total_db_operations']}")
            if 'avg_db_time' in stats:
                print(f"Average DB operation time: {stats['avg_db_time']:.2f}s")
                print(f"Total DB time: {stats['total_db_time']:.2f}s")
        
        print("="*50)
