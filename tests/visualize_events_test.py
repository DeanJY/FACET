import os
import click
import numpy as np

from tqdm import tqdm
from EvEye.utils.dvs_common_utils.representation.Histgram import HistgramBuilder
from EvEye.utils.dvs_common_utils.representation.TimeSurface import TimeSurfaceBuilder
from EvEye.utils.dvs_common_utils.representation.FrameStack import FrameStackBuilder

from EvEye.utils.dvs_common_utils.base.EventsIterator import EventsIterator
from EvEye.utils.processor.TxtProcessor import TxtProcessor
from EvEye.utils.visualization.visualization import save_image, save_batch_images


@click.command()
@click.option(
    "--file_path",
    type=str,
    default="/mnt/data2T/junyuan/eye-tracking/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_1/events/events.txt",
)
@click.option(
    "--output_path",
    type=str,
    default="/mnt/data2T/junyuan/eye-tracking/outputs/visualize_events_test_framestack_causal_linear_1",
)
def main(file_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    txt_processor = TxtProcessor(file_path)
    event_data = txt_processor.load_events_from_txt()
    frame_gap_us = 400000
    frame_interval_us = 400000

    # start_time = event_data["t"].min()
    # end_time = event_data["t"].max()
    start_time = 1657710786177477 - 40000
    end_time = 1657710988937640 + 40000
    duration = end_time - start_time
    with open(os.path.join(output_path, "time.txt"), "w") as time_file:
        time_file.write(
            f"Start Time: {start_time} us\nEnd Time: {end_time} us\nDuration: {duration} us\n"
        )

    events_iterator = EventsIterator(event_data, frame_gap_us, start_time)

    print("Toatal time of the events:", events_iterator.total_time, "us")
    height, width = 260, 346

    # histgram_builder = HistgramBuilder(height, width, frame_interval_us, max_count=10)
    # timesurface_builder = TimeSurfaceBuilder(height, width, frame_interval_us)
    framestack_builder = FrameStackBuilder(height, width, 10, frame_interval_us)

    for index, events in enumerate(
        tqdm(events_iterator, desc="Processing events", total=len(events_iterator))
    ):
        # print("----- New event buffer! -----")
        if events.size == 0:
            # print("No events to process in this buffer.")
            continue
        start_time = events_iterator.current_time - frame_gap_us

        # histgram = histgram_builder(events, start_time
        # histgram_vis = HistgramBuilder.visualize(histgram)
        # save_image(histgram_vis, f"{output_path}/{index}.png")
        # timesurface = timesurface_builder(events, start_time)
        # timesurface_vis = TimeSurfaceBuilder.visualize(timesurface)
        # save_image(timesurface_vis, f"{output_path}/{index}.png")
        framestack = framestack_builder(
            events,
            start_time,
            temporal_downsample=frame_interval_us,
            mode="causal_linear",
            max_count=10,
        )
        framestack_vis = FrameStackBuilder.visualize(framestack)
        save_batch_images(framestack_vis, f"{output_path}/{index:04}")


if __name__ == "__main__":
    main()
