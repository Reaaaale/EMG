BAUDRATE = 460800
CHANNEL_COUNT = 16
MIN_LIVE_INFERENCE_WINDOW_SHIFT = 100
REGISTER_NAMES = [
    "ID",
    "CONF1",
    "CONF2",
    "CONF3",
    "LOFF",
    "CH1",
    "CH2",
    "CH3",
    "CH4",
    "CH5",
    "CH6",
    "CH7",
    "CH8",
    "BIASp",
    "BIASn",
    "LOFFp",
    "LOFFn",
    "LOFFflip",
    "LOFFstatp",
    "LOFFstatn",
    "GPIO",
    "MISC1",
    "MISC2",
    "CONF4",
]
INPUT_MODES = ["NORMAL", "SHORT", "RLD", "TEST", "TEMP", "SUPPLY"]
PGA_VALUES = ["1", "2", "4", "6", "8", "12", "24", "Do not Use"]
FREQUENCIES = {
    "16kSPS": 16000,
    "8kSPS": 8000,
    "4kSPS": 4000,
    "2kSPS": 2000,
    "1kSPS": 1000,
    "500SPS": 500,
    "250SPS": 250,
}
