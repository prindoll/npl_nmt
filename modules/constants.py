# # DESIGNATE constants values for config
# DEFAULT_DECODE_STRATEGY = "BeamSearch"
# DEFAULT_STRATEGY_KWARGS = {}
# DEFAULT_SEED = 101
# DEFAULT_BATCH_SIZE = 64
# DEFAULT_EVAL_BATCH_SIZE = 8
# DEFAULT_TRAIN_TEST_SPLIT = 0.8
# DEFAULT_DEVICE = "cpu"
# DEFAULT_K = 5
# DEFAULT_INPUT_MAX_LENGTH = 200
# DEFAULT_MAX_LENGTH = 150
# DEFAULT_TRAIN_MAX_LENGTH = 100
# DEFAULT_LOWERCASE = True
# DEFAULT_NUM_KEEP_MODEL_TRAIN = 5
# DEFAULT_NUM_KEEP_MODEL_BEST = 5
# DEFAULT_SOS = "<sos>"
# DEFAULT_EOS = "<eos>"
# DEFAULT_PAD = "<pad>"
DEFAULT_DECODE_STRATEGY = "TopKSampling"
DEFAULT_STRATEGY_KWARGS = {
    "k": 50,  # Number of top tokens to sample from
    "temperature": 0.7  # Temperature to control randomness
}
DEFAULT_SEED = 101
DEFAULT_BATCH_SIZE = 64
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_TRAIN_TEST_SPLIT = 0.8
DEFAULT_DEVICE = "cuda"  # Updated to match YAML
DEFAULT_INPUT_MAX_LENGTH = 512  # Updated to match YAML
DEFAULT_MAX_LENGTH = 160  # Updated to match YAML
DEFAULT_TRAIN_MAX_LENGTH = 50  # Updated to match YAML
DEFAULT_LOWERCASE = False  # Updated to match YAML
DEFAULT_NUM_KEEP_MODEL_TRAIN = 5
DEFAULT_NUM_KEEP_MODEL_BEST = 5
DEFAULT_SOS = "<sos>"
DEFAULT_EOS = "<eos>"
DEFAULT_PAD = "<pad>"