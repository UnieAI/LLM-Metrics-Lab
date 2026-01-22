"""API Monitor Visualization Script

Generates comprehensive visualization of API performance metrics including:
- Token throughput (tokens/sec)
- Total tokens (cumulative)
- Prompt vs completion tokens
- Active/completed sessions
- Request success/failure rates
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns
import fire


class APIMetricsVisualizer:
    """Visualize API metrics from JSONL log file"""

    def __init__(self, log_file):
        self.log_file = log_file
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def read_log_data(self):
        """Read metrics from JSONL log file"""
        metrics = {
            'timestamps': [],
            'chars_per_second': [],
            'total_chars': [],
            'tokens_per_second': [],
            'total_tokens': [],
            'active_sessions': [],
            'completed_sessions': [],
            'successful_requests': [],
            'failed_requests': [],
            'prompt_tokens': [],
            'completion_tokens': [],
            'has_token_data': False,
        }

        with open(self.log_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                metrics['timestamps'].append(datetime.fromisoformat(data['timestamp']))
                metrics['chars_per_second'].append(data['chars_per_second'])
                metrics['total_chars'].append(data['total_chars'])
                metrics['active_sessions'].append(data['active_sessions'])
                metrics['completed_sessions'].append(data.get('completed_sessions', 0))
                metrics['successful_requests'].append(data.get('successful_requests', 0))
                metrics['failed_requests'].append(data.get('failed_requests', 0))
                metrics['prompt_tokens'].append(data.get('prompt_tokens', 0))
                metrics['completion_tokens'].append(data.get('completion_tokens', 0))

                # Handle token data with backward compatibility
                if 'tokens_per_second' in data:
                    metrics['has_token_data'] = True
                    metrics['tokens_per_second'].append(data['tokens_per_second'])
                    metrics['total_tokens'].append(data.get('total_tokens', 0))
                else:
                    metrics['tokens_per_second'].append(0)
                    metrics['total_tokens'].append(0)

        return metrics

    def create_visualization(self, output_file='api_metrics.png'):
        """Generate simplified metrics visualization (key metrics only)"""
        metrics = self.read_log_data()
        
        num_plots = 3 if metrics['has_token_data'] else 2
        fig, axes = plt.subplots(
            num_plots, 1,
            figsize=(15, 5 * num_plots),
            height_ratios=[1] * num_plots
        )
        
        # Handle single plot case (axes won't be array)
        if num_plots == 1:
            axes = [axes]
        
        fig.suptitle('API Performance Metrics', fontsize=16, y=0.995)

        plot_idx = 0

        if metrics['has_token_data']:
            # Plot 1: Tokens per Second (Throughput)
            axes[plot_idx].plot(metrics['timestamps'], metrics['tokens_per_second'], 
                               linewidth=3, marker='o', markersize=6, color='#2ecc71')
            axes[plot_idx].set_title('Tokens per Second (Throughput)', fontsize=14, fontweight='bold')
            axes[plot_idx].set_ylabel('Tokens/s', fontsize=12)
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].fill_between(metrics['timestamps'], metrics['tokens_per_second'], alpha=0.3, color='#2ecc71')
            plot_idx += 1

            # Plot 2: Total Tokens (Cumulative)
            axes[plot_idx].plot(metrics['timestamps'], metrics['total_tokens'], 
                               linewidth=3, marker='o', markersize=6, color='#3498db')
            axes[plot_idx].set_title('Total Tokens (Cumulative)', fontsize=14, fontweight='bold')
            axes[plot_idx].set_ylabel('Total Tokens', fontsize=12)
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].fill_between(metrics['timestamps'], metrics['total_tokens'], alpha=0.3, color='#3498db')
            plot_idx += 1
        else:
            # Fallback: Characters per Second
            axes[plot_idx].plot(metrics['timestamps'], metrics['chars_per_second'], 
                               linewidth=3, marker='o', markersize=6)
            axes[plot_idx].set_title('Characters per Second (Throughput)', fontsize=14, fontweight='bold')
            axes[plot_idx].set_ylabel('Chars/s', fontsize=12)
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].fill_between(metrics['timestamps'], metrics['chars_per_second'], alpha=0.3)
            plot_idx += 1

        # Plot: Success/Failure Rate (last plot for all)
        axes[plot_idx].plot(metrics['timestamps'], metrics['successful_requests'], 
                           linewidth=3, marker='o', markersize=6, label='Successful', color='#27ae60')
        axes[plot_idx].plot(metrics['timestamps'], metrics['failed_requests'], 
                           linewidth=3, marker='x', markersize=8, label='Failed', color='#e74c3c')
        axes[plot_idx].set_title('Request Success/Failure', fontsize=14, fontweight='bold')
        axes[plot_idx].set_ylabel('Count', fontsize=12)
        axes[plot_idx].legend(loc='best', fontsize=11)
        axes[plot_idx].grid(True, alpha=0.3)
        
        self._format_axes(axes)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _format_axes(self, axes):
        """Format all axes with consistent styling"""
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.tick_params(axis='x', rotation=45)

def main(log_file="api_monitor.jsonl", output_file="api_metrics.png"):
    """Main entry point for visualization"""
    print(f"Log File: {log_file}")
    print(f"Output File: {output_file}")

    visualizer = APIMetricsVisualizer(log_file)
    visualizer.create_visualization(output_file)
    print(f"✓ Visualization saved: {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
