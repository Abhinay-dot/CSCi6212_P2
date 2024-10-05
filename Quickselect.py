import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Quick Select with Median of Medians Method
def quick_select(arr, left, right, k):
    if k > 0 and k <= right - left + 1:
        n = right - left + 1
        medians = []

        # Divide the array into groups of 5 and find medians
        for i in range(0, n, 5):
            sub_left = left + i
            sub_right = min(sub_left + 4, right)
            median = find_median(arr, sub_left, sub_right)
            medians.append(median)

        median_of_medians = (medians[0] if len(medians) == 1 else
                             quick_select(medians, 0, len(medians) - 1, len(medians) // 2))

        partition_index = partition(arr, left, right, median_of_medians)

        if partition_index - left == k - 1:
            return arr[partition_index]
        if partition_index - left > k - 1:
            return quick_select(arr, left, partition_index - 1, k)
        return quick_select(arr, partition_index + 1, right, k - partition_index + left - 1)
    return float('inf')


def partition(arr, left, right, pivot):
    pivot_index = arr.index(pivot)
    arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
    partition_index = left

    for i in range(left, right):
        if arr[i] <= pivot:
            arr[i], arr[partition_index] = arr[partition_index], arr[i]
            partition_index += 1

    arr[partition_index], arr[right] = arr[right], arr[partition_index]
    return partition_index


def find_median(arr, left, right):
    sub_array = sorted(arr[left:right + 1])
    return sub_array[len(sub_array) // 2]


def generate_random_array(size):
    return np.random.randint(0, 10000, size).tolist()


def measure_execution_times():
    input_sizes = [15,135, 135*9, 135*81, 135*729]
    execution_times = []
    theoretical_times = []

    for size in input_sizes:
        arr = generate_random_array(size)
        start_time = time.time()
        quick_select(arr, 0, len(arr) - 1, size // 2)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1e9  # Convert to nanoseconds
        execution_times.append(execution_time)
        theoretical_time = size  # O(n) for theoretical comparison
        theoretical_times.append(theoretical_time)
        print(f"Execution time for input size {size} is {execution_time} ns")

    # Calculate the scaling constant
    scaling_constant = np.mean([execution / theoretical for execution, theoretical in zip(execution_times, theoretical_times)])
    print(f"Scaling constant: {scaling_constant}")

    # Prepare data for table
    scaled_times = [scaling_constant * theoretical for theoretical in theoretical_times]
    data = {
        "Input Size": input_sizes,
        "Execution Time (ns)": execution_times,
        "Theoretical Time (ns)": theoretical_times,
        "Scaled Time (ns)": scaled_times
    }
    df = pd.DataFrame(data)
    print(df)

    return input_sizes, execution_times, theoretical_times, scaled_times


def plot_graph(input_sizes, execution_times, theoretical_times, scaled_times):
    plt.figure(figsize=(12, 8))
    plt.plot(input_sizes, execution_times, marker='o', linestyle='-', color='b', label='Experimental Time')
    plt.plot(input_sizes, scaled_times, marker='o', linestyle='-.', color='g', label='Scaled Time')
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (ns)')
    plt.title('Quick Select Execution Times (Experimental vs Theoretica)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    input_sizes, execution_times, theoretical_times, scaled_times = measure_execution_times()
    plot_graph(input_sizes, execution_times, theoretical_times, scaled_times)
