# Evaluation of Structured Lyric Generation by LLMs

## Environment

Install the requirements as follows

```bash
conda create -n glrc python=3.8
conda activate glrc
pip install -r requirements.txt
```

## One-Shot Generation

We utilize a one-shot approach to prompt Large Language Models (LLMs) to generate structured lyrics.

An example of a prompt is as follows:

```
你将扮演一位流行音乐领域的专业作词人，根据[歌词主旨]和[乐段结构]对应生成一篇严格符合要求的歌词，比如我们假设：
  [歌词主旨]是
    "我想表达回忆、遗憾和时间流逝的深刻感慨。试图忘记过去的情感纠葛，但又无法真正释怀。对往昔美好时光的记忆也暗示着这些记忆带来的痛苦，希望接受命运并随着时间慢慢淡忘过去。"
  [乐段结构]是
    "(verse)\ncccccc cccccc\ncccccc ccR\ncccccc cccccR\ncccccccccR\n(verse)\ncccccc cccccR\ncccccc ccc\ncccccc cccccR\nccccccccc\n(chorus)\ncccccc cccccc\nccccccccR\nccccccccccR\nccccccccR\n(chorus)\ncccccc cccccR\nccccccccR\ncccccc cccccR\ncccccc ccR"
  则最终输出的歌词应该类似于
    "(verse)\n就这样忘记吧 怎么能忘记呢\n墨绿色的纠缠 你的他\n窗前流淌的歌 枕上开过的花\n岁月的风带它去了哪啊\n(verse)\n就这样忘记吧 怎么能忘记呢\n昏黄色的深情 你的他\n指尖燃起的火 喉头咽下的涩\n瞳孔里的星辰在坠落\n(chorus)\n总有些遗憾吗 总有些遗憾吧\n光阴它让纯粹蒙了灰\n如此蒂固根深又摇摇欲坠\n倒影中的轮廓他是谁\n(chorus)\n你也是这样吗你 也是这样吧\n目送了太久忘了出发\n说不出是亏欠 等不到是回答\n就这样老去吧 老去吧"

即要求在符合[歌词主旨]的前提下，严格按照[乐段结构]中给出的格式生成歌词；其中表示乐段名称的'(verse)'、'(chorus)'，和表示换行的'\n'、停顿的' '空格等这些信息保持不变，剩下每一行里的每个c和行尾的R代表需要生成替换的一个字符，R不同与c的是，它的生成替换字必须在同一个乐段里押韵！

请严格按照以上要求输出最终结果，不必生成中间思考和推理过程。

歌词主旨：%s
乐段结构：%s
```

## Structure Similarity Calculation

After the LLM generates structured lyrics, the structural similarity between the generated lyrics and actual lyrics can be calculated using `glrc_obj_eval.py`.

For example, in `glrc_obj_eval.py`,

```python
o_msstr = '(verse)\ncccccccc\nccccccR\ncccccccc\nccccccR\n(verse)\ncccccccccR\nccccccccc\ncccccccccR\nccccccccc\n(chorus)\nccccccc\nccccccccR\nccccR\nccccccccR\n(chorus)\nccccccR\nccccccccR\nccccR\nccccccccc\n(verse)\ncccccccc\nccccccR\ncccccccc\nccccccR\n(verse)\ncccccccccR\nccccccccc\ncccccccccR\nccccccccc\n(chorus)\nccccccc\nccccccccR\nccccR\nccccccccR\n(chorus)\nccccccR\nccccccccR\nccccR\nccccccccc(verse)\ncccccccc\nccccccR\ncccccccc\nccccccR\n(verse)\ncccccccccR\nccccccccc\ncccccccccR\nccccccccc' # The actual structure in the prompt

gen_lrc = '(verse)\n如果一切可以重新来过\n我会在相遇的街口\n把你紧握再也不放手\n不会再让你远走\n(verse)\n如果上天能再给我一次\n爱你的机会不会辜负\n也许这次机会会更加\n更加珍惜不再错过你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每天都能见到你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每一分都为你而活\n(others)\n如果一切可以重新来过\n我会在相遇的街口\n把你紧握再也不放手\n不会再让你远走\n(verse)\n如果上天能再给我一次\n爱你的机会不会辜负\n也许这次机会会更加\n更加珍惜不再错过你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每天都能见到你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每一分都为你而活\n(verse)\n如果一切可以重新来过\n我会在相遇的街口\n把你紧握再也不放手\n不会再让你远走\n(verse)\n如果上天能再给我一次\n爱你的机会不会辜负\n也许这次机会会更加\n更加珍惜不再错过你' # The generated structured lyrics by LLMs
```

Run

```bash
python glrc_obj_eval.py
```

And then get the score

```
phase1   score:  6.4309, p1ws_sr=0.64309392
phase2.1 score: 19.5000, acmp_sr=0.60000000 (am_p1sr is False)
phase2.2 score: 10.4978, acmp_sr=0.59987503
phase3   score:  6.2195, wcm_amr=0.51840281
phase4   score: 11.5531, ttrc_sr=0.96296296
extra    score:  2.0196, r_ratio=0.54166667 (pll_rc=10)
*FNL-SCORE-SUM: 56.22101801738358
```