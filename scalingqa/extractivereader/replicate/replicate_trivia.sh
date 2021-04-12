#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
cd ../../../
python3 -m scalingqa.extractivereader.run_extractive_reader_filter $DIR/replicate_trivia_filtr.py > $DIR/replicate_trivia_filtr.log
python3 -m scalingqa.extractivereader.run_extractive_reader_train $DIR/replicate_trivia.py