import nidaqmx as daq

with daq.Task() as task:

    task.ai_channels.add_ai_voltage_chan("Dev2/ai0")

    print(task.read(number_of_samples_per_channel=5))