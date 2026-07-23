"""texture-frames-de — German frame-semantic parser (gbert encoder, SALSA corpus).

    from texture_frames_de import FrameParser
    parser = FrameParser()                       # downloads weights on first use
    for ann in parser.parse("Die Regierung erhöhte die Steuern ."):
        print(ann.frame, ann.trigger, ann.arguments)
"""
from .pipeline import Argument, FrameAnnotation, FrameParser

__all__ = ["FrameParser", "FrameAnnotation", "Argument"]
__version__ = "0.2.0"
