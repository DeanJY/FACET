import os
import re
import click
import numpy as np
import natsort
import itertools

from tqdm import tqdm
from EvEye.utils.dvs_common_utils.representation.Histgram import HistgramBuilder
from EvEye.utils.dvs_common_utils.representation.TimeSurface import TimeSurfaceBuilder
from EvEye.utils.dvs_common_utils.representation.FrameStack import FrameStackBuilder

from EvEye.utils.dvs_common_utils.base.EventsIterator import EventsIterator
from EvEye.utils.processor.TxtProcessor import TxtProcessor
from EvEye.utils.visualization.visualization import save_image, save_batch_images

base_path = "/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset/train"
ouptut_base_path = "/mnt/data2T/junyuan/eye-tracking/outputs/EventsFrames"


def parse_args(base_path):
    user_list = []
    side_list = []
    session_list = []
    files = natsort.natsorted([f for f in os.listdir(base_path) if f.endswith(".txt")])
    for file in files:
        match = re.match(r"(user\d+)_(left|right)_(session_\d+\_\d+\_\d+)_", file)
        if match:
            user_list.append(match.group(1))
            side_list.append(match.group(2))
            session_list.append(match.group(3))
    return user_list, side_list, session_list


def get_start_time(base_path, user, side, session):
    file_path = os.path.join(base_path, f"{user}_{side}_{session}_centers.txt")
    with open(file_path, "r") as file:
        first_line = file.readline()
    start_time = first_line.split(",")[0]
    print(f"Start time of {user}_{side}_{session}: {start_time}")
    return start_time


def main(base_path, ouptut_base_path):
    centers_path = base_path + "/label"
    events_path = base_path + "/data"
    user_list, side_list, session_list = parse_args(centers_path)
    iter_zip = itertools.islice(zip(user_list, side_list, session_list), 69, None)
    for user, side, session in iter_zip:
        start_time = get_start_time(centers_path, user, side, session)
        event_path = os.path.join(events_path, f"{user}_{side}_{session}_events.txt")
        output_path = os.path.join(ouptut_base_path, f"{user}_{side}_{session}")
        os.makedirs(output_path, exist_ok=True)
        txt_processor = TxtProcessor(event_path)
        event_data = txt_processor.load_events_from_txt()
        frame_gap_us = 400000
        frame_interval_us = 400000

        # start_time = event_data["t"].min()
        # end_time = event_data["t"].max()
        start_time_flag = True
        if int(start_time) < event_data["t"].max():
            start_time = int(start_time) - 40000
            start_time_flag = True
        else:
            start_time = event_data["t"].min()
            start_time_flag = False

        end_time = event_data["t"].max()
        duration = end_time - start_time

        if start_time_flag:
            with open(os.path.join(output_path, "time.txt"), "w") as time_file:
                time_file.write(
                    f"Start Time (Same to frame): {start_time} us\nEnd Time: {end_time} us\nDuration: {duration} us\n"
                )
        else:
            with open(os.path.join(output_path, "time.txt"), "w") as time_file:
                time_file.write(
                    f'Start Time (event_data["t"].min()): {start_time} us\nEnd Time: {end_time} us\nDuration: {duration} us\n'
                )
            with open(
                "/mnt/data2T/junyuan/eye-tracking/Recording.txt", "a"
            ) as recording_file:
                recording_file.write(f"{user}_{side}_{session}\n")

        events_iterator = EventsIterator(event_data, frame_gap_us, start_time)

        print("Toatal time of the events:", events_iterator.total_time, "us")
        if start_time_flag:
            print("Start time is the same as the frame.")
        else:
            print("Start time is the same as event_data['t'].min().")
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
            # start_time = events_iterator.current_time - frame_gap_us
            start_time = events_iterator.current_time

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
                max_count=1,
            )
            framestack_vis = FrameStackBuilder.visualize(framestack)
            save_batch_images(framestack_vis, f"{output_path}/{index:04}")


if __name__ == "__main__":
    main(base_path, ouptut_base_path)
