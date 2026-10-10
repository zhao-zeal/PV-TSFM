from layers.chronosx.input_injection_block import InputInjectionBlock
from layers.chronosx.output_injection_block import OutputInjectionBlock

injection_blocks_map = {
    "IIB": (InputInjectionBlock, None),
    "OIB": (None, OutputInjectionBlock),
    "IIB+OIB": (InputInjectionBlock, OutputInjectionBlock),
}
