# rm /nfs/music-5-test/jukebox/model/s0/*
# rm /nfs/music-5-test/jukebox/model/s1/*
# rm /nfs/music-5-test/mert95/model/s0/*
# rm /nfs/music-5-test/mert95/model/s1/*
# rm /nfs/music-5-test/mert300/model/s0/*
# rm /nfs/music-5-test/mert300/model/s1/*
# rm /nfs/music-5-test/music2vec/model/s0/*
# rm /nfs/music-5-test/music2vec/model/s1/*
# rm /nfs/music-5-test/encodec/model/s0/*
# rm /nfs/music-5-test/encodec/model/s1/*


screen -S mert330-s0 python3 train/mert330-s0.py
screen -S mert330-s1 python3 train/mert330-s1.py
screen -S mert95-s0 python3 train/mert95-s0.py
screen -S mert95-s1 python3 train/mert95-s1.py
screen -S music2vec-s0 python3 train/music2vec-s0.py
screen -S music2vec-s1 python3 train/music2vec-s1.py
screen -S encodec-s0 python3 train/encodec-s0.py
screen -S encodec-s1 python3 train/encodec-s1.py
screen -S jukebox-s0 python3 train/jukebox-s0.py
screen -S jukebox-s1 python3 train/jukebox-s1.py