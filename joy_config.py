import os
from .uitls import read_json_file

joy_base_path = os.path.dirname(os.path.realpath(__file__))
joy_config = read_json_file(os.path.join(joy_base_path, "joy_config.json"))