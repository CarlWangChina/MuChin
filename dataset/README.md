# MuChin 1k Dataset

## Music Description Annotation

The "question" within the description JSON file of each data are denoted by an ID format, and the mapping relationships of questions, subquestions and options with respect to their IDs are depicted in the files `qidmap.json`, `subqidmap.json` and `opidmap.json`, respectively.

`amat_desc_json` is the JSON format file containing amateur descriptions.

`prof_desc_json` is the JSON format file containing professional descriptions.


## Musical Structure Annotation

`str_lyric` contains the lyrics and their musical segment structure information.

`str_rhyme` indicates the positions of rhymes within the lyrics.
 
`tknz_json` provides both the musical segment information and timestamp data for each line of lyrics. The timestamps in tknz_json are incorrect, but the musical segments are correct (after manually annotating the musical segments, we hoped to match timestamps from raw_lyric, but found that we couldn't match them because during the manual annotation of musical segments, many musical phrases were manually merged or split, leading to an inconsistent number of lines). If training requires timestamps, it's better to directly use the timestamps from the `raw_lyric`.

`raw_lyric`provides the lyric text and timestamp text. Compared to tknz_json, the timestamps here are correct (timestamps exported from the KTV file), but the timestamps in raw_lyric also have some time offset issues not caused by this work. Timestamps are not the focus of this dataset; 

the focus of the manual annotation for this dataset is on the following four aspects: colloquial description, professional description, musical segment labels, and rhymes.


## Music Audio Files

Song audio cannot be used for commercial model training without the authorization of the copyright holder. Audio download link: https://pan.baidu.com/s/1D4xGQhYUwWbpaHyAS71dfw?pwd=1234 Extract password: 1234

Currently, a subset of the MuChin dataset comprising 1,000 instances has been made publicly available as open-source within this repository. The dataset features comprehensive coverage of both amateur and professional descriptions, as well as intricate structural metadata such as musical sections and rhyme structures.

We invite scholars and researchers to employ this resource broadly in their research initiatives. Proper reference to its use in academic publications is appreciated.

`meta_info` : we have provided metadata in the files `meta_info` that includes the song names, artist names, album names, and release years (with some information potentially missing) for the annotated songs. 

`list-fp.enlarged_matches.jsonl` lists 724 songs from this set that are duplicates of the enlarged 6066 version of muchin.
