"""Entry point: text -> aligned 2D and 3D motion.

    python infer_text_2d3d.py --text "a person walks forward and waves"

All logic lives in the ``movt`` package; this file only keeps the familiar
command name at the project root. See ``movt/pipeline.py`` for the pipeline.
"""

from movt.cli import main

if __name__ == "__main__":
    main()
