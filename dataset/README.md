# MuChin Dataset

## Music Description Annotation

The "question" within the description JSON file of each data are denoted by an ID format, and the mapping relationships of questions, subquestions and options with respect to their IDs are depicted in the files `qidmap.json`, `subqidmap.json` and `opidmap.json`, respectively.

`amat_desc_json` is the JSON format file containing amateur descriptions.

`prof_desc_json` is the JSON format file containing professional descriptions.

## Musical Structure Annotation

`str_lyric` contains the lyrics and their musical segment structure information.

`str_rhyme` indicates the positions of rhymes within the lyrics.

`tknz_json` provides both the musical segment information and timestamp data for each line of lyrics.

`raw_lyric` is in the original LRC format, with timestamp information for each line of lyrics.

## Acquisition of Music Audio Files
`meta_info` : Due to the restrictions on the use and copyright of audio playback, we have provided metadata in the files `meta_info` that includes the song names, artist names, album names, and release years (with some information potentially missing) for the annotated songs. 
