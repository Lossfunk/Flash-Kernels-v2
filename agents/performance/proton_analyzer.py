import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import tempfile
import os

@dataclass
class DeviceSpecs:
    """Device specifications for roofline analysis"""
    name: str
    compute_capability: Tuple[int, int]
    multi_processor_count: int
    max_threads_per_multiprocessor: int
    max_shared_memory_per_multiprocessor: int
    max_shared_memory_per_block: int
    warp_size: int
    max_threads_per_block: int
    memory_clock_rate: int  # kHz
    memory_bus_width: int   # bits
    l2_cache_size: int      # bytes
    total_memory: int       # bytes
    
    # Computed properties
    peak_memory_bandwidth: float  # GB/s
    peak_flops_fp32: float       # GFLOPS
    peak_flops_fp16: float       # GFLOPS
    peak_flops_tensor: float     # GFLOPS (tensor cores if available)
    roofline_slope_fp32: float   # FLOPS/byte
    roofline_slope_fp16: float   # FLOPS/byte
    
    @classmethod
    def from_device(cls, device: str = "cuda") -> 'DeviceSpecs':
        """Create DeviceSpecs from current CUDA device"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
            
        props = torch.cuda.get_device_properties(device)
        
        # Get device-specific estimates for missing properties
        estimates = cls._get_device_estimates(props.name, props.major, props.minor)
        
        # Use estimates for memory bandwidth since PyTorch doesn't expose memory specs
        peak_bw = estimates['memory_bandwidth_gb_s']
        
        # Estimate peak FLOPS based on compute capability and device estimates
        base_clock_ghz = estimates['base_clock_ghz']
        cores_per_sm = cls._get_cores_per_sm(props.major, props.minor)
        
        peak_fp32 = props.multi_processor_count * cores_per_sm * base_clock_ghz
        peak_fp16 = peak_fp32 * 2  # FP16 can be 2x faster
        
        # Use device-specific tensor core estimates
        peak_tensor = estimates['tensor_flops_gflops']
        
        # Safely get attributes that might not be available in all PyTorch versions
        warp_size = getattr(props, 'warp_size', 32)  # Default to 32 for NVIDIA GPUs
        l2_cache_size = getattr(props, 'L2_cache_size', 0)  # Default to 0 if not available
        
        return cls(
            name=props.name,
            compute_capability=(props.major, props.minor),
            multi_processor_count=props.multi_processor_count,
            max_threads_per_multiprocessor=props.max_threads_per_multi_processor,
            max_shared_memory_per_multiprocessor=estimates['shared_memory_per_sm'],
            max_shared_memory_per_block=estimates['shared_memory_per_block'],
            warp_size=warp_size,
            max_threads_per_block=estimates['max_threads_per_block'],
            memory_clock_rate=estimates['memory_clock_rate_khz'],
            memory_bus_width=estimates['memory_bus_width_bits'],
            l2_cache_size=l2_cache_size,
            total_memory=props.total_memory,
            peak_memory_bandwidth=peak_bw,
            peak_flops_fp32=peak_fp32,
            peak_flops_fp16=peak_fp16,
            peak_flops_tensor=peak_tensor,
            roofline_slope_fp32=peak_fp32 / peak_bw,
            roofline_slope_fp16=peak_fp16 / peak_bw,
        )
    
    @staticmethod
    def _get_cores_per_sm(major: int, minor: int) -> int:
        """Get CUDA cores per SM based on compute capability"""
        # Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
        if major == 2:
            return 32
        elif major == 3:
            return 192
        elif major == 5:
            return 128
        elif major == 6:
            return 64 if minor == 0 else 128
        elif major == 7:
            return 64
        elif major == 8:
            return 64
        elif major == 9:
            return 128
        else:
            return 64  # Conservative fallback
    
    @staticmethod
    def _get_device_estimates(device_name: str, major: int, minor: int) -> Dict[str, Any]:
        """Get device-specific estimates for properties not exposed by PyTorch"""
        
        # Default estimates based on compute capability
        estimates = {
            'memory_bandwidth_gb_s': 500.0,  # Conservative default
            'base_clock_ghz': 1.5,
            'memory_clock_rate_khz': 6000000,  # 6 GHz effective
            'memory_bus_width_bits': 256,
            'shared_memory_per_sm': 164 * 1024,  # 164KB for modern GPUs
            'shared_memory_per_block': 48 * 1024,  # 48KB typical
            'max_threads_per_block': 1024,
            'tensor_flops_gflops': 0.0
        }
        
        # Device-specific overrides based on known specifications
        device_lower = device_name.lower()
        
        # RTX 40 series
        if 'rtx 4090' in device_lower:
            estimates.update({
                'memory_bandwidth_gb_s': 1008.0,
                'base_clock_ghz': 2.2,
                'tensor_flops_gflops': 165000.0,  # ~165 TFLOPS
                'shared_memory_per_sm': 100 * 1024,
            })
        elif 'rtx 4080' in device_lower:
            estimates.update({
                'memory_bandwidth_gb_s': 717.0,
                'base_clock_ghz': 2.2,
                'tensor_flops_gflops': 120000.0,
            })
        elif 'rtx 4070' in device_lower or 'rtx 4060' in device_lower or 'rtx 4050' in device_lower:
            estimates.update({
                'memory_bandwidth_gb_s': 400.0,  # Varies by model
                'base_clock_ghz': 2.0,
                'tensor_flops_gflops': 80000.0,  # Approximate
            })
        
        # RTX 30 series
        elif 'rtx 3090' in device_lower:
            estimates.update({
                'memory_bandwidth_gb_s': 936.0,
                'base_clock_ghz': 1.7,
                'tensor_flops_gflops': 142000.0,
            })
        elif 'rtx 3080' in device_lower:
            estimates.update({
                'memory_bandwidth_gb_s': 760.0,
                'base_clock_ghz': 1.7,
                'tensor_flops_gflops': 119000.0,
            })
        elif 'rtx 3070' in device_lower:
            estimates.update({
                'memory_bandwidth_gb_s': 448.0,
                'base_clock_ghz': 1.7,
                'tensor_flops_gflops': 101000.0,
            })
        
        # Data center GPUs
        elif 'a100' in device_lower:
            estimates.update({
                'memory_bandwidth_gb_s': 1935.0,
                'base_clock_ghz': 1.4,
                'tensor_flops_gflops': 312000.0,
                'shared_memory_per_sm': 164 * 1024,
            })
        elif 'h100' in device_lower:
            estimates.update({
                'memory_bandwidth_gb_s': 3350.0,
                'base_clock_ghz': 1.8,
                'tensor_flops_gflops': 989000.0,
                'shared_memory_per_sm': 228 * 1024,
            })
        elif 'v100' in device_lower:
            estimates.update({
                'memory_bandwidth_gb_s': 900.0,
                'base_clock_ghz': 1.4,
                'tensor_flops_gflops': 125000.0,
                'shared_memory_per_sm': 96 * 1024,
            })
        
        # Adjust based on compute capability
        if major >= 8:  # Ampere and newer
            estimates['shared_memory_per_sm'] = max(estimates['shared_memory_per_sm'], 164 * 1024)
            estimates['max_threads_per_block'] = 1024
        elif major == 7:  # Volta/Turing
            estimates['shared_memory_per_sm'] = max(estimates['shared_memory_per_sm'], 96 * 1024)
            estimates['max_threads_per_block'] = 1024
        
        return estimates

@dataclass
class KernelMetrics:
    """Metrics collected from kernel execution"""
    name: str
    runtime_ms: float
    flops: float
    bytes_transferred: float
    arithmetic_intensity: float
    achieved_bandwidth_gb_s: float
    achieved_flops_gflops: float
    occupancy_percent: float
    bottleneck: str  # "memory_bound" or "compute_bound"
    efficiency_percent: float
    roofline_position: Tuple[float, float]  # (intensity, performance)
    raw_proton_metrics: Optional[Dict[str, float]] = None # Added for detailed proton output
    profile_hatchet_path: Optional[str] = None # Path to the .hatchet file
    yaml_report_path: Optional[str] = None # Path to the YAML report if generated

import triton
import triton.profiler as proton
import triton.profiler.viewer as proton_viewer
import json
import time
from contextlib import contextmanager

class ProtonRooflineAnalyzer:
    """Roofline analysis using Triton's Proton profiler"""
    
    def __init__(self, device: str = "cuda"):
        self.device_specs = DeviceSpecs.from_device(device)
        self.profile_data = {}
        
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        return {
            "device_name": self.device_specs.name,
            "compute_capability": f"{self.device_specs.compute_capability[0]}.{self.device_specs.compute_capability[1]}",
            "multiprocessors": self.device_specs.multi_processor_count,
            "max_threads_per_sm": self.device_specs.max_threads_per_multiprocessor,
            "shared_memory_per_sm_kb": self.device_specs.max_shared_memory_per_multiprocessor // 1024,
            "shared_memory_per_block_kb": self.device_specs.max_shared_memory_per_block // 1024,
            "warp_size": self.device_specs.warp_size,
            "memory_bandwidth_gb_s": self.device_specs.peak_memory_bandwidth,
            "peak_flops_fp32_gflops": self.device_specs.peak_flops_fp32,
            "peak_flops_fp16_gflops": self.device_specs.peak_flops_fp16,
            "peak_tensor_flops_gflops": self.device_specs.peak_flops_tensor,
            "total_memory_gb": self.device_specs.total_memory / (1024**3),
            "l2_cache_mb": self.device_specs.l2_cache_size / (1024**2),
        }
    
    @contextmanager
    def profile_context(self, profile_name: str):
        """Context manager for Proton profiling"""
        proton.start(profile_name, hook="triton")
        try:
            yield
        finally:
            proton.finalize()
    
    def profile_kernel(self, 
                      kernel_func: Callable, 
                      args: Tuple,
                      kernel_name: str,
                      flops: float,
                      bytes_transferred: float,
                      warmup_iters: int = 10,
                      profile_iters: int = 100,
                      output_dir: Optional[str] = None) -> KernelMetrics:
        """Profile a kernel and return comprehensive metrics"""
        
        # Warmup
        for _ in range(warmup_iters):
            kernel_func(*args)
        torch.cuda.synchronize()
        
        profile_name = f"{kernel_name}_profile"
        
        # Determine profiling directory context
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            # Use a context manager-like structure for chdir even without TemporaryDirectory
            class DirContext:
                def __init__(self, new_path):
                    self.new_path = new_path
                    self.old_path = None
                def __enter__(self):
                    self.old_path = os.getcwd()
                    os.chdir(self.new_path)
                    return self.new_path
                def __exit__(self, exc_type, exc_val, exc_tb):
                    os.chdir(self.old_path)
            
            profiling_context_manager = DirContext(output_dir)
            hatchet_file_path_in_context = f"{profile_name}.hatchet" # Will be in output_dir
        else:
            # Use TemporaryDirectory if no output_dir is specified
            # tempfile.TemporaryDirectory returns the path to the temp dir
            profiling_context_manager = tempfile.TemporaryDirectory() 
            # In this case, hatchet_file_path_in_context is relative to the temp_dir,
            # so _parse_proton_results will be called with "profile_name.hatchet"
            # while CWD is the temp_dir.
            hatchet_file_path_in_context = f"{profile_name}.hatchet"

        detailed_metrics = {}
        runtime_ms = 0.0

        with profiling_context_manager as current_profiling_dir:
            # If output_dir was used, current_profiling_dir is output_dir.
            # If tempfile was used, current_profiling_dir is the path to the temp dir.
            # The CWD is already changed to current_profiling_dir by the context managers.
            
            try:
                with self.profile_context(profile_name): # profile_name doesn't include path here
                    start_time = time.perf_counter()
                    
                    for _ in range(profile_iters):
                        kernel_func(*args)
                    
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                
                runtime_ms = (end_time - start_time) * 1000 / profile_iters
                
                # The hatchet file (e.g., "kernel_name_profile.hatchet") is expected to be in current_profiling_dir
                # which is the CWD at this point.
                # If output_dir is provided, we need the full path for _parse_proton_results if it's called outside this CWD scope,
                # but since _parse_proton_results itself just needs a filename if it's in CWD, this is fine.
                # The hatchet_file_path_in_context is just the filename.
                detailed_metrics = self._parse_proton_results(hatchet_file_path_in_context)
                
                if output_dir:
                    print(f"Proton profile data, including '{hatchet_file_path_in_context}', saved to: {os.path.abspath(current_profiling_dir)}")

            except Exception as e:
                print(f"An error occurred during profiling or parsing: {e}")
                # Ensure we still have a valid runtime_ms if timing succeeded before an error
                if 'end_time' in locals() and 'start_time' in locals() and profile_iters > 0:
                     runtime_ms = (end_time - start_time) * 1000 / profile_iters
                else:
                     runtime_ms = -1 # Indicate error or incomplete profiling
                # detailed_metrics will remain empty or as it was before error
            # CWD is automatically restored by DirContext or TemporaryDirectory's __exit__

        # Calculate derived metrics
        arithmetic_intensity = flops / bytes_transferred if bytes_transferred > 0 else 0
        achieved_bandwidth = bytes_transferred / (runtime_ms / 1000) / 1e9  # GB/s
        achieved_flops = flops / (runtime_ms / 1000) / 1e9  # GFLOPS
        
        # Determine bottleneck
        if arithmetic_intensity < self.device_specs.roofline_slope_fp32:
            bottleneck = "memory_bound"
            theoretical_peak = self.device_specs.peak_memory_bandwidth
            achieved_peak = achieved_bandwidth
        else:
            bottleneck = "compute_bound"
            theoretical_peak = self.device_specs.peak_flops_fp32
            achieved_peak = achieved_flops
        
        efficiency = (achieved_peak / theoretical_peak) * 100 if theoretical_peak > 0 else 0
        
        # Estimate occupancy (simplified)
        occupancy = min(100.0, efficiency * 1.2)  # Rough estimate
        
        return KernelMetrics(
            name=kernel_name,
            runtime_ms=runtime_ms,
            flops=flops,
            bytes_transferred=bytes_transferred,
            arithmetic_intensity=arithmetic_intensity,
            achieved_bandwidth_gb_s=achieved_bandwidth,
            achieved_flops_gflops=achieved_flops,
            occupancy_percent=occupancy,
            bottleneck=bottleneck,
            efficiency_percent=efficiency,
            roofline_position=(arithmetic_intensity, achieved_flops),
            raw_proton_metrics=detailed_metrics,
            profile_hatchet_path=os.path.join(os.getcwd(), hatchet_file_path_in_context) if not output_dir else os.path.join(output_dir, hatchet_file_path_in_context), # Store full path
        )
    
    def _parse_proton_results(self, hatchet_file: str) -> Dict[str, float]:
        """Parse Proton hatchet file for detailed metrics"""
        
        if not os.path.exists(hatchet_file):
            print(f"Warning: Could not access Proton results file: {hatchet_file}")
            return {}

        # List of potential metrics to try individually.
        # Based on user feedback from inspecting the .hatchet file directly.
        metrics_to_try = [
            "time (ns)",     # Confirmed available from user's .hatchet file content
            # "time",          # Alias or simplified name, not the direct key in the file
            # "time/ms",
            # "time_ms",
            # "duration/ms", 
            # "duration",
            # "gflops",
            # "tflop/s", 
            # "tflops", 
            # "flops",
            # "bytes_transferred",
            # "bytes", 
            # "bandwidth",
            # "achieved_occupancy",
            # "occupancy", 
            # "sm_efficiency",
            # "count" # Also present, but might not be directly used for performance metrics in the same way as time.
        ]
        
        collected_metrics = {}
        found_any_metric = False
        problematic_metrics_log = [] # To store specific "metric not found" errors if user wants to see them later

        for metric_key in metrics_to_try:
            try:
                # Attempt to parse this single metric.
                # proton_viewer.parse returns (tree, metrics_dict)
                # metrics_dict will contain {metric_key: value} if successful for *this specific key*.
                _, parsed_metric_value_dict = proton_viewer.parse([metric_key], hatchet_file)
                
                if metric_key in parsed_metric_value_dict and parsed_metric_value_dict[metric_key] is not None:
                    # Sanitize key for dictionary (e.g., replace '/') and prefix
                    safe_key = f"proton_{metric_key.replace('/', '_').replace('-', '_')}"
                    collected_metrics[safe_key] = parsed_metric_value_dict[metric_key]
                    found_any_metric = True
            except Exception as e:
                # Check if the error is the common "Metric ... is not found."
                if "is not found" in str(e).lower() and metric_key in str(e):
                    problematic_metrics_log.append(f"Notice: Metric '{metric_key}' was not found in {hatchet_file}.")
                else:
                    # Log other types of errors encountered during parsing of a specific metric
                    problematic_metrics_log.append(f"Warning: Issue parsing metric '{metric_key}' from {hatchet_file}: {e}")
        
        if not found_any_metric:
            # Attempt manual JSON parsing as fallback
            try:
                import json
                with open(hatchet_file, 'r') as f:
                    data = json.load(f)
                # data is expected to be a list; first element has children list
                def traverse(node):
                    results = []
                    if isinstance(node, dict):
                        if 'metrics' in node and isinstance(node['metrics'], dict):
                            results.append(node['metrics'])
                        for key in ('children',):
                            if key in node and isinstance(node[key], list):
                                for child in node[key]:
                                    results.extend(traverse(child))
                    elif isinstance(node, list):
                        for item in node:
                            results.extend(traverse(item))
                    return results
                all_metrics_dicts = traverse(data)
                # Look for keys 'time (ns)' and 'count'
                total_time_ns = 0
                total_count = 0
                for md in all_metrics_dicts:
                    if 'time (ns)' in md:
                        total_time_ns += md['time (ns)']
                    if 'count' in md:
                        total_count += md['count']
                if total_time_ns > 0:
                    collected_metrics['proton_time_ns_manual'] = total_time_ns
                    found_any_metric = True
                if total_count > 0:
                    collected_metrics['proton_count_manual'] = total_count
                if found_any_metric:
                    print("Manual fallback: extracted metrics from hatchet JSON without proton_viewer.parse.")
            except Exception as manual_e:
                print(f"Manual JSON parsing of hatchet file failed: {manual_e}")
        
        if not found_any_metric:
            # This message prints if none of the metrics in metrics_to_try were successfully parsed.
            print(f"Warning: No detailed metrics successfully parsed from Proton file: {hatchet_file}.")
            # Optionally, print the detailed log of problematic metrics if debugging is needed
            # for problem_log_entry in problematic_metrics_log:
            #     print(problem_log_entry)
            print("This could be normal if the profiler isn't configured to emit these specific metrics.")
            print("Falling back to primary timing for performance analysis.")
        elif problematic_metrics_log and any("Warning:" in log for log in problematic_metrics_log):
            # If we found some metrics, but also had some warnings (not just "not found")
            print(f"Notice: Some detailed metrics parsed from {hatchet_file}, but issues were encountered with others:")
            for problem_log_entry in problematic_metrics_log:
                 if "Warning:" in problem_log_entry: # Only print actual warnings, not "not found" notices
                    print(f"  - {problem_log_entry}")

        return collected_metrics
    
    def analyze_roofline_position(self, metrics: KernelMetrics) -> Dict[str, Any]:
        """Analyze kernel position on roofline model"""
        analysis = {
            "kernel_name": metrics.name,
            "arithmetic_intensity": metrics.arithmetic_intensity,
            "achieved_performance_gflops": metrics.achieved_flops_gflops,
            "bottleneck": metrics.bottleneck,
            "efficiency_percent": metrics.efficiency_percent,
            "recommendations": []
        }
        
        # Generate specific recommendations based on roofline position
        if metrics.bottleneck == "memory_bound":
            analysis["recommendations"].extend([
                "Kernel is memory-bound - focus on reducing DRAM traffic",
                "Consider operator fusion to reuse data in shared memory",
                "Optimize memory access patterns for coalescing",
                "Use vectorized loads/stores where possible",
                f"Current bandwidth utilization: {metrics.achieved_bandwidth_gb_s:.1f}/{self.device_specs.peak_memory_bandwidth:.1f} GB/s"
            ])
            
            if metrics.efficiency_percent < 50:
                analysis["recommendations"].append("Low memory efficiency - check for uncoalesced accesses")
                
        else:  # compute_bound
            analysis["recommendations"].extend([
                "Kernel is compute-bound - focus on computational efficiency",
                "Check SM occupancy and register usage",
                "Consider using tensor cores for mixed precision",
                "Optimize instruction-level parallelism",
                f"Current compute utilization: {metrics.achieved_flops_gflops:.1f}/{self.device_specs.peak_flops_fp32:.1f} GFLOPS"
            ])
            
            if self.device_specs.peak_flops_tensor > 0:
                analysis["recommendations"].append(
                    f"Consider tensor cores for up to {self.device_specs.peak_flops_tensor:.0f} GFLOPS"
                )
        
        # Occupancy recommendations
        if metrics.occupancy_percent < 50:
            analysis["recommendations"].extend([
                "Low occupancy detected - consider:",
                "- Reducing register usage per thread",
                "- Reducing shared memory usage per block",
                "- Adjusting block size and grid dimensions"
            ])
        
        return analysis
    
    def generate_optimization_report(self, metrics: KernelMetrics, output_dir: Optional[str] = None) -> str:
        """Generate a comprehensive optimization report and save to YAML"""
        analysis = self.analyze_roofline_position(metrics)
        device_info = self.get_device_info()
        
        report_data = {
            "device_information": device_info,
            "kernel_metrics": {
                "name": metrics.name,
                "runtime_ms": metrics.runtime_ms,
                "arithmetic_intensity_flop_byte": metrics.arithmetic_intensity,
                "achieved_bandwidth_gb_s": metrics.achieved_bandwidth_gb_s,
                "achieved_performance_gflops": metrics.achieved_flops_gflops,
                "bottleneck": metrics.bottleneck.upper(),
                "efficiency_percent": metrics.efficiency_percent,
                "estimated_occupancy_percent": metrics.occupancy_percent,
                "flops_total": metrics.flops,
                "bytes_transferred_total": metrics.bytes_transferred
            },
            "roofline_analysis": {
                "position_intensity_gflops": (metrics.arithmetic_intensity, metrics.achieved_flops_gflops),
                "roofline_slope_fp32_flop_byte": self.device_specs.roofline_slope_fp32
            },
            "optimization_recommendations": analysis["recommendations"],
            "raw_proton_metrics": metrics.raw_proton_metrics if metrics.raw_proton_metrics else {},
            "profile_hatchet_path": metrics.profile_hatchet_path
        }

        # report_str = f"""\nKERNEL PERFORMANCE ANALYSIS REPORT...""" # Original string too long, will reconstruct from report_data

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            yaml_filename = f"{metrics.name}_perf_report.yaml"
            yaml_path = os.path.join(output_dir, yaml_filename)
            try:
                with open(yaml_path, 'w') as f:
                    # Using json.dump for simplicity, can be replaced by PyYAML if specific YAML features are needed
                    json.dump(report_data, f, indent=2) 
                metrics.yaml_report_path = yaml_path # Set the path on the metrics object
                print(f"Performance report saved to: {yaml_path}")
            except Exception as e:
                print(f"Error saving performance report to YAML: {e}")
                metrics.yaml_report_path = None # Explicitly set to None on failure
        else:
            metrics.yaml_report_path = None # No output_dir, so no YAML path
        
        # Construct the string report (as before, but using report_data for consistency)
        report_str = f"""
KERNEL PERFORMANCE ANALYSIS REPORT
{'='*50}

Device Information:
- GPU: {report_data['device_information']['device_name']}
- Compute Capability: {report_data['device_information']['compute_capability']}
- Multiprocessors: {report_data['device_information']['multiprocessors']}
- Peak Memory Bandwidth: {report_data['device_information']['memory_bandwidth_gb_s']:.1f} GB/s
- Peak FP32 Performance: {report_data['device_information']['peak_flops_fp32_gflops']:.1f} GFLOPS
- Peak FP16 Performance: {report_data['device_information']['peak_flops_fp16_gflops']:.1f} GFLOPS

Kernel Metrics:
- Name: {report_data['kernel_metrics']['name']}
- Runtime: {report_data['kernel_metrics']['runtime_ms']:.3f} ms
- Arithmetic Intensity: {report_data['kernel_metrics']['arithmetic_intensity_flop_byte']:.2f} FLOP/byte
- Achieved Bandwidth: {report_data['kernel_metrics']['achieved_bandwidth_gb_s']:.1f} GB/s
- Achieved Performance: {report_data['kernel_metrics']['achieved_performance_gflops']:.1f} GFLOPS
- Bottleneck: {report_data['kernel_metrics']['bottleneck']}
- Efficiency: {report_data['kernel_metrics']['efficiency_percent']:.1f}%
- Estimated Occupancy: {report_data['kernel_metrics']['estimated_occupancy_percent']:.1f}%

Roofline Analysis:
- Position: ({report_data['roofline_analysis']['position_intensity_gflops'][0]:.2f}, {report_data['roofline_analysis']['position_intensity_gflops'][1]:.1f})
- Roofline Slope (FP32): {report_data['roofline_analysis']['roofline_slope_fp32_flop_byte']:.2f} FLOP/byte

Optimization Recommendations:
"""        
        for i, rec in enumerate(report_data["optimization_recommendations"], 1):
            report_str += f"{i}. {rec}\n"
        
        if report_data["raw_proton_metrics"]:
            report_str += "\nRaw Proton Metrics:\n"
            for k, v in report_data["raw_proton_metrics"].items():
                report_str += f"- {k}: {v}\n"
        if report_data["profile_hatchet_path"]:
            report_str += f"\nProton Profile Data: {report_data['profile_hatchet_path']}\n"

        return report_str

class MemoryAccessOptimizer:
    """Optimize memory access patterns based on device capabilities"""
    
    def __init__(self, device_specs: DeviceSpecs):
        self.device_specs = device_specs
        self.cache_line_size = 128  # bytes
        
    def analyze_access_pattern(self, 
                             tensor_shape: Tuple[int, ...], 
                             dtype: torch.dtype,
                             access_stride: int = 1) -> Dict[str, Any]:
        """Analyze memory access pattern efficiency"""
        
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_bytes = torch.tensor(tensor_shape).prod().item() * element_size
        
        # Calculate coalescing efficiency
        elements_per_cache_line = self.cache_line_size // element_size
        coalescing_efficiency = min(1.0, elements_per_cache_line / access_stride)
        
        # Vectorization opportunities
        vector_widths = {
            torch.float32: 4,  # 128-bit vectors
            torch.float16: 8,
            torch.bfloat16: 8,
            torch.int32: 4,
            torch.int16: 8,
            torch.int8: 16,
        }
        
        max_vector_width = vector_widths.get(dtype, 1)
        
        return {
            "total_bytes": total_bytes,
            "element_size": element_size,
            "coalescing_efficiency": coalescing_efficiency,
            "max_vector_width": max_vector_width,
            "cache_lines_accessed": total_bytes // self.cache_line_size,
            "recommendations": self._generate_access_recommendations(
                coalescing_efficiency, max_vector_width, access_stride
            )
        }
    
    def _generate_access_recommendations(self, 
                                       coalescing_eff: float, 
                                       vector_width: int,
                                       stride: int) -> List[str]:
        """Generate memory access optimization recommendations"""
        recommendations = []
        
        if coalescing_eff < 0.8:
            recommendations.append(
                f"Poor coalescing efficiency ({coalescing_eff:.1%}) - "
                f"consider reducing access stride (current: {stride})"
            )
        
        if vector_width > 1:
            recommendations.append(
                f"Use vectorized loads/stores with width {vector_width} for better bandwidth"
            )
        
        recommendations.extend([
            "Align data structures to cache line boundaries",
            "Consider data layout transformations (AoS vs SoA)",
            "Use shared memory for data reuse patterns"
        ])
        
        return recommendations 