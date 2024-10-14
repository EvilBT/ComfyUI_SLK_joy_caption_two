from .joy_caption_two_node import Joy_caption_two
from .joy_caption_two_node import Joy_caption_two_advanced
from .joy_caption_two_node import Batch_joy_caption_two_advanced
from .joy_caption_two_node import Batch_joy_caption_two
from .joy_caption_two_node import Joy_caption_two_load
from .joy_caption_two_node import Joy_extra_options

NODE_CLASS_MAPPINGS = {
    "Joy_caption_two": Joy_caption_two,
    "Joy_caption_two_advanced": Joy_caption_two_advanced,
    "Batch_joy_caption_two": Batch_joy_caption_two,
    "Batch_joy_caption_two_advanced": Batch_joy_caption_two_advanced,
    "Joy_caption_two_load": Joy_caption_two_load,
    "Joy_extra_options": Joy_extra_options,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Joy_caption_two": "Joy Caption Two",
    "Joy_caption_two_advanced": "Joy Caption Two Advanced",
    "Batch_joy_caption_two": "Batch Joy Caption Two",
    "Batch_joy_caption_two_advanced": "Batch Joy Caption Two Advanced",
    "Joy_caption_two_load": "Joy Caption Two Load",
    "Joy_extra_options": "Joy Caption Extra Options",
}