from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot


def benchmark(devices):
    '''Benchmark each device by computing matrix products'''
    times = {device: [] for device in devices}
    sizes = range(100, 700, 500)

    for size in sizes:

        print(f"Calculating {size}x{size} matrix product")

        for device in devices:

            shape = (size, size)
            data_type = tf.float32
            with tf.device(device):
                mat1 = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
                mat2 = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
                matmul = tf.matmul(mat1, mat2)

            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
                start_time = time.time()
                result = session.run(matmul)
                time_taken = time.time() - start_time
                print(f"{device} took {round(time_taken,2)}s")
                times[device].append(time_taken)

    return times, sizes


def plot_results(devices, sizes, times):
    '''Plot the benchmark results'''
    fig, (ax1, ax2) = pyplot.subplots(2, 1, sharex=True)
    
    for device in devices:
        ax1.plot(sizes, times[device], 'o-', label=device)
    ax1.set_ylabel('Compute Time')
    ax1.set_title('Device Compute Time vs. Matrix size')
    ax1.legend(devices, loc=2)
    
    ax2.plot(sizes, np.divide(times[devices[1]], times[devices[0]]), 'o-', label=device)
    ax2.set_ylabel('Speedup')
    ax2.set_xlabel('Matrix size')
    ax2.set_title('Speedup GPU relative to CPU vs. Matrix size')
    
    pyplot.show()


def experiment():
    '''Run an experiment that compares CPU and GPU device performance'''
    devices = ["/gpu:0", "/cpu:0"]
    times, sizes = benchmark(devices)
    plot_results(devices, sizes, times)

experiment()
