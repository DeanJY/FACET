import numpy as np

from dv_processing import Accumulator
from dv_processing import EventStore
from dv_processing import Frame

class DvAccumulator(object):
    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480), 
        decay_function = Accumulator.Decay.EXPONENTIAL, 
        decay_param: float = 1.0e6,
        min_potential: float = 0.0, 
        max_potential: float = 1.0,
        neutral_potential: float = 0.5,
        event_contribution: float = 0.15, 
        ignore_polarity: bool = False,
        synchronous_decay: bool = False
    ) -> None:
        self._accumulator = Accumulator(resolution)
        self._accumulator.setMinPotential(min_potential)
        self._accumulator.setMaxPotential(max_potential)
        self._accumulator.setNeutralPotential(neutral_potential)
        self._accumulator.setEventContribution(event_contribution)
        self._accumulator.setDecayFunction(decay_function)
        self._accumulator.setDecayParam(decay_param)
        self._accumulator.setIgnorePolarity(ignore_polarity)
        self._accumulator.setSynchronousDecay(synchronous_decay)
        
    @property
    def accumulator(self) -> Accumulator:
        return self._accumulator
    
    def clear(self):
        self._accumulator.clear()
        
    def accept(self, event_chunk: EventStore):
        self._accumulator.accept(event_chunk)
        
    def generate_frame(self) -> Frame:
        return self._accumulator.generateFrame()
    
    def generate_np_frame_e2e(self, event_chunk: EventStore) -> np.ndarray:
        self.clear()
        self.accept(event_chunk)
        return self.generate_frame().image