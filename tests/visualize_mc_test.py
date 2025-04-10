import click
import numpy as np
import cv2
from metavision_core.event_io import EventsIterator
from dvs_common_utils.representations.time_surface import TimeSurfaceBuilder
from dvs_common_utils.representations.histgram import HistgramBuilder
from dvs_common_utils.processor.motion_compensation import MotionCompensation2D


@click.command()
@click.option("--file_path", type=str, default="/data/data/test/1.raw")
def main(file_path: str):
    frame_gap_us = 10000
    frame_interval_us = 1000
    mv_iterator = EventsIterator(
        input_path=file_path,
        delta_t=frame_gap_us,
        start_ts=0 * frame_gap_us,
    )
    height, width = mv_iterator.get_size()
    time_surface_builder = TimeSurfaceBuilder(height, width, frame_interval_us)
    histgram_builder = HistgramBuilder(
        height, width, interval_us=frame_interval_us, max_count=5
    )
    motion_compensation_2d = MotionCompensation2D(height, width, t_flag=True)
    for evs in mv_iterator:
        print("----- New event buffer! -----")
        if evs.shape[0] == 0:
            continue
        start_time_us = mv_iterator.current_time - frame_gap_us
        evs_mc = motion_compensation_2d(evs, start_time_us, vx=10000)

        frame = time_surface_builder(evs, start_time_us)
        frame_vis = TimeSurfaceBuilder.visualize(time_surface=frame)

        # Note that compensated events can not be used to build time surface as they have already lost time information
        # frame = time_surface_builder(evs_mc, start_time_us)
        # frame_vis = TimeSurfaceBuilder.visualize(frame)
        frame_mc = histgram_builder(evs_mc, start_time_us)
        frame_mc_vis = HistgramBuilder.visualize(frame_mc)

        cv2.imshow("x", np.concatenate([frame_vis, frame_mc_vis], axis=1))
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
            # return
        # print(time_surface.shape)


if __name__ == "__main__":
    main()
