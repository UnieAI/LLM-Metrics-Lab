import subprocess
import sys
import re
import argparse

def convert_benchmark_result(benchmark_text: str) -> tuple[dict, list]:
    """
    Parses a serving benchmark result string and converts it into a
    specified dictionary structure.

    Args:
        benchmark_text: A string containing the benchmark results.

    Returns:
        A tuple containing a dictionary with the benchmark data in the target
        format and a list of the keys in the desired order.
    """
    # Define all keys for the final format to ensure order and completeness
    target_keys = [
        'concurrency', 'total requests', 'elapsed_time (second)',
        'number of prompt tokens', 'number of decode tokens',
        'prompt tokens throughput (token/s)', 'decode tokens throughput (token/s)',
        'total tokens throughput (token/s)', 'RPS (request per second)',
        'RPM (request per minute)', 'TTFT (time to first token(second))',
        'P50 (second)', 'P99 (second)', 'TBT (token latency in second)',
        'TPS (token/s per request)', 'TPS decode (token/s per request)',
        'TPS prompt (token/s per request)'
    ]

    # Initialize the result dict with empty strings
    converted_result = {key: '' for key in target_keys}

    # Dictionary to map original keys to the new keys.
    # The keys here must exactly match the text in the benchmark output.
    key_mapping = {
        'Successful requests': 'total requests',
        'Maximum request concurrency': 'concurrency',
        'Benchmark duration (s)': 'elapsed_time (second)',
        'Total input tokens': 'number of prompt tokens',
        'Total generated tokens': 'number of decode tokens',
        'Request throughput (req/s)': 'RPS (request per second)',
        'Output token throughput (tok/s)': 'decode tokens throughput (token/s)',
        'Total Token throughput (tok/s)': 'total tokens throughput (token/s)',
        'Mean TTFT (ms)': 'TTFT (time to first token(second))',
        'Median TTFT (ms)': 'P50 (second)',
        'P99 TTFT (ms)': 'P99 (second)',
        'Mean ITL (ms)': 'TBT (token latency in second)',
    }

    raw_data = {}
    # This simpler regex correctly captures the full key and the value.
    pattern = re.compile(r'^(.*?):\s+([\d\.]+)$')

    for line in benchmark_text.strip().split('\n'):
        # Skip lines that are not data lines (headers, footers, separators)
        if '---' in line or '===' in line or not line.strip():
            continue
        
        match = pattern.match(line.strip())
        if match:
            # The key is everything before the colon, with trailing spaces removed.
            key = match.group(1).strip()
            value_str = match.group(2)
            if key:
                value = float(value_str) if '.' in value_str else int(value_str)
                raw_data[key] = value

    # Map the raw data to the new structure
    for old_key, new_key in key_mapping.items():
        if old_key in raw_data:
            value = raw_data[old_key]
            # Convert ms to seconds if the original key indicates milliseconds
            if '(ms)' in old_key:
                value /= 1000.0
            converted_result[new_key] = value

    # --- Calculate additional fields ---
    if converted_result.get('RPS (request per second)'):
        converted_result['RPM (request per minute)'] = converted_result['RPS (request per second)'] * 60
    
    # Calculate prompt throughput first
    if raw_data.get('Total input tokens') and raw_data.get('Benchmark duration (s)', 0) > 0:
        prompt_throughput = raw_data['Total input tokens'] / raw_data['Benchmark duration (s)']
        converted_result['prompt tokens throughput (token/s)'] = prompt_throughput

    # Get values needed for the new calculations, with safety checks
    decode_throughput = converted_result.get('decode tokens throughput (token/s)')
    prompt_throughput = converted_result.get('prompt tokens throughput (token/s)')
    total_requests = converted_result.get('total requests')
    concurrency = converted_result.get('concurrency')
    total_token_throughput = converted_result.get('total tokens throughput (token/s)')

    # New calculation for TPS (token/s per request)
    if all(isinstance(v, (int, float)) for v in [total_token_throughput, total_requests, concurrency]) and total_requests > 0 and concurrency > 0:
        converted_result['TPS (token/s per request)'] = total_token_throughput / total_requests

    # New calculation for TPS decode (token/s per request)
    if all(isinstance(v, (int, float)) for v in [decode_throughput, total_requests, concurrency]) and total_requests > 0 and concurrency > 0:
        converted_result['TPS decode (token/s per request)'] = decode_throughput / total_requests
    
    # New calculation for TPS prompt (token/s per request)
    if all(isinstance(v, (int, float)) for v in [prompt_throughput, total_requests, concurrency]) and total_requests > 0 and concurrency > 0:
        converted_result['TPS prompt (token/s per request)'] = prompt_throughput / total_requests


    return converted_result, target_keys

def print_formatted_table(data: dict, key_order: list):
    """Prints the final data in the specified table format."""
    print("|########################################|")
    max_key_length = max(len(key) for key in key_order)
    for key in key_order:
        value = data.get(key, '')
        # Format float values for consistent display
        if isinstance(value, float):
            print(f"{key+':':<{max_key_length+1}} {value: >15.5f}")
        else:
            print(f"{key+':':<{max_key_length+1}} {str(value): >15}")

def run_and_parse_benchmark(command: list):
    """
    Runs a benchmark command, captures the relevant output in real-time,
    and parses it into the desired format.
    """
    benchmark_output_lines = []
    is_capturing = False

    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            print(f"--- Running benchmark: {' '.join(command)} ---")
            print("--- Live Output ---")
            for line in proc.stdout:
                sys.stdout.write(line)
                if "============ Serving Benchmark Result ============" in line:
                    is_capturing = True
                if is_capturing:
                    benchmark_output_lines.append(line.strip())
                if "==================================================" in line and is_capturing:
                    break
        
        if not benchmark_output_lines:
            print("\nError: Benchmark result block was not found in the script's output.")
            return

        benchmark_text_to_parse = "\n".join(benchmark_output_lines)
        
        print("\n--- Parsing Captured Result ---")
        parsed_data, key_order = convert_benchmark_result(benchmark_text_to_parse)
        
        '''
        # Log the raw benchmark text to a file named after the concurrency
        concurrency = parsed_data.get('concurrency', 'unknown_concurrency')
        log_filename = f"{concurrency}.log"
        try:
            with open(log_filename, 'w') as f:
                f.write(benchmark_text_to_parse)
            print(f"\nBenchmark text logged to '{log_filename}'")
        except IOError as e:
            print(f"\nCould not write log file: {e}")
        '''

        print("\n--- Final Formatted Result ---")
        print_formatted_table(parsed_data, key_order)

    except FileNotFoundError:
        print(f"\nError: Command not found. Make sure '{command[0]}' is correct and in your system's PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

# --- HOW TO USE ---
# Run this script from your terminal and pass the benchmark command and its arguments directly.
#
# Example:
# python this_script_name.py python benchmark_serving.py --model your-model --tokenizer path/to/tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a serving benchmark script, capture its output, and format the results.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example usage:
  python %(prog)s python benchmark_serving.py --model my-model --tokenizer my-tokenizer
"""
    )
    
    parser.add_argument(
        'command',
        nargs=argparse.REMAINDER, # Captures all following arguments into a list
        help="The benchmark command to execute."
    )

    args = parser.parse_args()

    if args.command:
        # If a command is provided, run it.
        run_and_parse_benchmark(args.command)
    else:
        # If no command is provided, show help and run a simulation for demonstration.
        print("No benchmark command provided. Displaying help and running a simulation.")
        parser.print_help()
        print("\n" + "="*50 + "\n")

        # The command below simulates the output of your benchmark script.
        simulation_command = [
            'python', '-c',
            """
import time, sys
print('Simulating benchmark start...')
time.sleep(0.5)
print('Simulating requests...')
time.sleep(0.5)
sys.stdout.write('''
============ Serving Benchmark Result ============
Successful requests:                     100
Maximum request concurrency:             1
Benchmark duration (s):                  2080.52
Total input tokens:                      4095900
Total generated tokens:                  204800
Request throughput (req/s):              0.05
Output token throughput (tok/s):         98.44
Total Token throughput (tok/s):          2067.13
---------------Time to First Token----------------
Mean TTFT (ms):                          1809.60
Median TTFT (ms):                        1811.61
P99 TTFT (ms):                           2242.35
P100 TTFT (ms):                          2247.58
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.28
Median TPOT (ms):                        9.27
P99 TPOT (ms):                           9.33
P100 TPOT (ms):                          9.34
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.28
Median ITL (ms):                         9.27
P99 ITL (ms):                            9.65
P100 ITL (ms):                           13.95
----------------End-to-end Latency----------------
Mean E2EL (ms):                          20804.68
Median E2EL (ms):                        20800.57
P99 E2EL (ms):                           21207.26
P100 E2EL (ms):                          21280.32
==================================================
''')
sys.stdout.flush()
print('\\nSimulating benchmark finish.')
"""
        ]
        run_and_parse_benchmark(simulation_command)

