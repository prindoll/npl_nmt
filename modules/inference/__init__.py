from modules.inference.decode_strategy import DecodeStrategy
from modules.inference.beam_search import BeamSearch
from modules.inference.prototypes import BeamSearch2
from modules.inference.sampling_temperature import GreedySearch
from modules.inference.topksampling import TopKSampling

strategies = {
        "BeamSearch": BeamSearch,
        "BeamSearch2": BeamSearch2,
        "GreedySearch": GreedySearch,
        "TopKSampling": TopKSampling,
}
