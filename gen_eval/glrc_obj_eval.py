"""Generated LRC Objective Evaluation."""

import re

import jieba
from pypinyin import Style, pinyin
from pypinyin_dict.phrase_pinyin_data import large_pinyin

large_pinyin.load()

from pypinyin_dict.pinyin_data import cc_cedict

cc_cedict.load()
# from pypinyin_dict.pinyin_data import kxhc1983
# kxhc1983.load()

# 中华新韵分韵表 https://baike.baidu.com/item/%E6%8A%BC%E9%9F%B5/192771#6
# '注音' -> '简名'
RHYTHM_TABLE = {
    'ㄚ': '01M', # 一麻: ['a', 'ia', 'ua']
    'ㄛ': '02B', # 二波: ['o', 'uo']
    'ㄜ': '03G', # 三歌: ['e']
    'ㄝ': '04J', # 四皆: ['ie', 've']

    ############## 五支: -i
    'ㄓ': '05Z', # ['zhi']
    'ㄔ': '05Z', # ['chi']
    'ㄕ': '05Z', # ['shi']
    'ㄖ': '05Z', # ['ri']
    'ㄗ': '05Z', # ['zi']
    'ㄘ': '05Z', # ['ci']
    'ㄙ': '05Z', # ['si']

    'ㄦ': '06E', # 六儿: ['er']
    'ㄧ': '07Q', # 七齐: ['i']
    'ㄟ': '08W', # 八微: ['ei', 'ui']
    'ㄞ': '09K', # 九开: ['ai', 'uai']
    'ㄨ': '10G', # 十姑: ['u']
    'ㄩ': '11Y', # 十一鱼: ['v']
    'ㄡ': '12H', # 十二侯: ['ou', 'iu']
    'ㄠ': '13H', # 十三豪: ['ao', 'iao']
    'ㄢ': '14H', # 十四寒: ['an', 'ian', 'uan', 'van']
    'ㄣ': '15H', # 十五痕: ['en', 'in', 'un', 'vn']
    'ㄤ': '16T', # 十六唐: ['ang', 'iang', 'uang']

    ############## 十七庚: eng, ing, ueng
    'ㄥ': '17G', # ['eng']
    'ㄧㄥ': '17G', # ['ing']
    'ㄨㄥ': '17G', # ['weng']

    ############## 十八东: ong, iong
    '+ㄨㄥ': '18D', # ['ong']
    'ㄩㄥ': '18D', # ['iong']
}


def mapping_rhythm_name(w_zhuyin: str):
    """Return value mapped from `RHYTHM_TABLE`, or word zhuyin without tone
    from the passed-in `w_zhuyin` if no any value can be mapped, e.g., empty
    `w_zhuyin` will get empty."""

    zy_notone = re.sub(r'[˙ˊˇˋ]', '', w_zhuyin) # remove 声调 first
    rhythm = zy_notone[-1:]
    rtname = RHYTHM_TABLE.get(rhythm)
    if not rtname: return rhythm
    if rhythm != 'ㄥ': return rtname

    rhythm = zy_notone[-2:]
    if 'ㄩㄥ' == rhythm:
        return RHYTHM_TABLE.get(rhythm, '18D')
    elif 'ㄧㄥ' == rhythm:
        return RHYTHM_TABLE.get(rhythm, '17G')
    elif 'ㄨㄥ' == rhythm:
        return RHYTHM_TABLE.get(rhythm, '17G') \
            if rhythm == zy_notone else \
            RHYTHM_TABLE.get('+'+rhythm, '18D')
    else:
        return RHYTHM_TABLE.get(rhythm[-1:], '17G')

def find_last_zh_word(line: str):
    """Return ('', -1) if not found any zh-cn character in whole line."""

    idx = len(line) - 1
    c = line[idx:]
    while (c < '\u4e00' or '\u9fff' < c) and idx > -1:
        idx -= 1
        c = line[idx:idx+1]
    return c, idx

def find_indices(pattern: str, text: str):
    """Return list of tuples, with matched start and end index for each."""

    matches = re.finditer(pattern, text)
    return [(m.start(), m.end()) for m in matches]


def split_repl_zhc2cr(sect_lrc: str, keepends=True):
    """"Split `sect_lrc` str to lines list and replace all zh-cn character to
    'c' or 'R' (for last 'c' if rhyme in the same rhythm to the other lines).

    return a 2 value tuple `(replaced_line_list, original_line_list)`"""

    # rhythm info of last zh-word in each line: {
    #   'rhythm_name': [
    #      (line_index, last_cn_word_index_in_line)
    #      ...
    #   ]
    # }, used for 'c' -> 'R' if they are the same rhythm
    last_zhw_rinfo = {}
    # line list to return
    rt_lines:list[str] = []
    og_lines = sect_lrc.splitlines(keepends)
    for i, line in enumerate(og_lines):
        _, last_zhw_idx = find_last_zh_word(line)
        phrases = list(jieba.cut(line[:last_zhw_idx+1]))
        zyl = pinyin(phrases, style=Style.BOPOMOFO, strict=False)
        # w_zhuyin = zyl[-1:][0] if len(zyl[-1:]) > 0 else ''
        # rhythm_name = mapping_rhythm_name(w_zhuyin)
        if len(zyl) > 0 and len(zyl[-1]) > 0:
            lc_zhuyin = zyl[-1][0]
        else:
            lc_zhuyin = ''
        rhythm_name = mapping_rhythm_name(lc_zhuyin)
        last_zhw_rinfo.setdefault(rhythm_name, [])
        last_zhw_rinfo[rhythm_name].append((i, last_zhw_idx,))
        # last_zhw_rns.append(mapping_rhythm_name(zyl[-1:][:1]))
        # only replace zh-cn character
        rt_lines.append(re.sub(r'[\u4e00-\u9fff]', 'c', line))
    for rn, infos in last_zhw_rinfo.items():
        if not rn or len(infos) < 2: continue
        for info in infos:
            il = info[0]; ci = info[1]
            tl = rt_lines[il]
            rt_lines[il] = tl[:ci] + 'R' + tl[ci+1:]
    return rt_lines, og_lines

_RP_MSNT = r'\([\w-]+\)\n?'

def cnv_lrc_2cmp_infos(llm_gen: str):
    """Convert LLM generated lrc string to comparable informations.

    return a tuple of 4 values (
        0. full abstracted lrc string,
        1. section infos dict list,
        2. section name code sequence,
        3. section line count sequence)."""

    msnt_indices = find_indices(_RP_MSNT, llm_gen)
    mindices_len = len(msnt_indices)
    # full abstracted lrc str to return
    rt_alrc = ''
    # section infos dict list to return
    rt_sidl = []
    # section name code sequence to return
    rt_sncs = ''
    # section line count sequence to return
    rt_slcs = ''
    for i, ct in enumerate(msnt_indices):
        if 0 == i and ct[0] > 0:
            rt_alrc += llm_gen[:ct[0]]
        msntl = llm_gen[ct[0]:ct[1]]
        sect = {'msntl': msntl}
        rt_alrc += msntl
        rt_sncs += msntl[1].upper()
        if (i+1) < mindices_len:
            nt = msnt_indices[i+1]
            sect_lrc = llm_gen[ct[1]:nt[0]]
        else:
            sect_lrc = llm_gen[ct[1]:]
        # abstracted and original section lrc lines
        abs_sll, o_lines = split_repl_zhc2cr(sect_lrc)
        sect['lines'] = abs_sll
        sect['og_ls'] = o_lines
        # abstracted section lrc str
        abs_slrc = ''.join(abs_sll)
        # sect['mslrc'] = abs_slrc
        rt_alrc += abs_slrc
        rt_sidl.append(sect)
        rt_slcs += chr(len(abs_sll)+48)
    return rt_alrc, rt_sidl, rt_sncs, rt_slcs

def cnv_mss_2cmp_infos(msstr: str):
    """Convert music structure string to comparable informations, similar to
    `cnv_mss_2cmp_infos`, return a tuple without the full abstracted-lrc part,
    but a total lrc-line count (section name lines excluded) instead."""

    msnt_indices = find_indices(_RP_MSNT, msstr)
    mindices_len = len(msnt_indices)
    rt_ttlc = 0
    rt_sidl = []
    rt_sncs = ''
    rt_slcs = ''
    for i, ct in enumerate(msnt_indices):
        msntl = msstr[ct[0]:ct[1]]
        sect = {'msntl': msntl}
        rt_sncs += msntl[1].upper()
        if (i+1) < mindices_len:
            nt = msnt_indices[i+1]
            sp = msstr[ct[1]:nt[0]]
        else:
            sp = msstr[ct[1]:]
        s_lines = sp.splitlines(True)
        sect['lines'] = s_lines
        l_count = len(s_lines)
        rt_ttlc += l_count
        rt_sidl.append(sect)
        rt_slcs += chr(l_count+48)
    return rt_ttlc, rt_sidl, rt_sncs, rt_slcs


# 10 50 20 20, extra 10
TOTAL_POINTS = 100
EXTRA_POINTS = 10
# Phase 1, whole sequences’ similarity
PH1_POINTS = TOTAL_POINTS * 0.10
# Phase 2:
#   - 2.1 section (count, name, order)
#   - 2.2 line count in each section
PH2_POINTS = TOTAL_POINTS * 0.50
# Phase 3, word count in each line
PH3_POINTS = TOTAL_POINTS * 0.20
# Phase 4, rhyme match + extra
PH4_POINTS = TOTAL_POINTS * 0.20

# Phase 2.1, points percent
PH21_P_PCT = 0.65
# Phase 2.2, points percent
PH22_P_PCT = 0.35

from difflib import HtmlDiff, SequenceMatcher, ndiff


def scoring(gen_lrc: str, o_msstr: str, am_p1sr=False):
    """Scoring the LLM generated lrc `gen_lrc` by the original music structure
    string, `o_msstr`, which is used for it's gen-prompt input.

    return a tuple of 3 values (final score of float, detailed log string,
    mss-diff HTML content)."""

    fnl_sum_score = 0.0
    detail_record = ''

    g_msstr, g_mssbl, g_snc_s, _ = cnv_lrc_2cmp_infos(gen_lrc)
    o_ttllc, o_mssbl, o_snc_s, _ = cnv_mss_2cmp_infos(o_msstr)

    detail_record += 'gen_lrc={!r}\no_msstr={!r}\n\ng_msstr={!r}\n\n'\
                    .format(gen_lrc, o_msstr, g_msstr)

    sm1 = SequenceMatcher(None, o_msstr, g_msstr)
    detail_record += "two abs-msstr sequences' overall comparison:\n"
    for tag, i1, i2, j1, j2 in sm1.get_opcodes():
        detail_record += '{:<9}o[{:<4}:{:>4}]  <->  g[{:<4}:{:>4}]\n'\
                        .format(tag, i1, i2, j1, j2)
        # detail_record += '    {!r}\n    ↓↓↓\n    {!r}\n'\
        #                 .format(o_msstr[i1:i2], g_msstr[j1:j2])

    sm21 = SequenceMatcher(None, o_snc_s, g_snc_s)
    # sum of max equaled section line count
    max_equ_slc_sum = 0
    # sum of line count by matched 2 sections, initialize it to 0.01
    # to decrease the final match ratio a little, and more than that
    # for avoiding the Db0 error.
    matched2s_lcsum = 0.01
    # valid word count matching ratio, accumulated-multiply
    # rhyming count in original/generated
    # paralleled rhyming count in both sides
    wcm_amr = 1.0; rc_ino = rc_ing = pll_rc = 0
    for m in sm21.get_matching_blocks():
        if 0 == m.size: break
        os_rg = range(m.a, m.a+m.size)
        gs_rg = range(m.b, m.b+m.size)
        for osi, gsi in zip(os_rg, gs_rg):
            # matched_2s_imap[osi] = gsi
            olines = o_mssbl[osi]['lines']
            glines = g_mssbl[gsi]['lines']
            oslc = len(olines)
            gslc = len(glines)
            min_slc = min(oslc, gslc)
            for x in range(0, min_slc):
                olx = olines[x].strip()
                glx = glines[x].strip()
                olrc = olx.count('R')
                glcc = glx.count('c')
                glsc = glx.count(' ')
                glrc = glx.count('R')
                rc_ino += olrc
                rc_ing += glrc
                if glrc & olrc: pll_rc += 1
                olxl = len(olx)
                min_lxcc = min(olxl, glcc+glsc+glrc)
                wcm_amr *= min_lxcc*2.0 / (olxl+len(glx))
            max_equ_slc_sum += min_slc
            matched2s_lcsum += oslc + gslc

    detail_record += "\nsection name code sequences' difference:\n"
    detail_record += ''.join(
        list(ndiff([o_snc_s+'\n'], [g_snc_s+'\n'], charjunk=None))
    )

    # whole abs-msstr sequences’ similarity ratio
    p1ws_sr = sm1.ratio()
    ph1score = PH1_POINTS * p1ws_sr
    fnl_sum_score += ph1score
    detail_record += '\nphase1   score:{:>8.4f}, p1ws_sr={:.8f}'\
                    .format(ph1score, p1ws_sr)

    # accumulated-multiply similarity ratio, effect all bellow phases
    acmp_sr = (p1ws_sr if am_p1sr else 1.0) * sm21.ratio()
    p21score = PH2_POINTS * PH21_P_PCT * acmp_sr
    fnl_sum_score += p21score
    detail_record += '\nphase2.1 score:{:>8.4f}, acmp_sr={:.8f}'\
        ' (am_p1sr is {})'.format(p21score, acmp_sr, am_p1sr)

    acmp_sr *= max_equ_slc_sum * 2.0 / matched2s_lcsum
    p22score = PH2_POINTS * PH22_P_PCT * acmp_sr
    fnl_sum_score += p22score
    detail_record += '\nphase2.2 score:{:>8.4f}, acmp_sr={:.8f}'\
                    .format(p22score, acmp_sr)

    ph3score = PH3_POINTS * wcm_amr * acmp_sr
    fnl_sum_score += ph3score
    detail_record += '\nphase3   score:{:>8.4f}, wcm_amr={:.8f}'\
                    .format(ph3score, wcm_amr)

    if (rc_ino + rc_ing) == 0:
        ttrc_sr = 0.0
    else:
        ttrc_sr = float(2*min(rc_ino, rc_ing) / (rc_ino+rc_ing))
    ph4score = PH4_POINTS * ttrc_sr * acmp_sr
    fnl_sum_score += ph4score
    detail_record += '\nphase4   score:{:>8.4f}, ttrc_sr={:.8f}'\
                    .format(ph4score, ttrc_sr)

    # extra score, based on rhyme match and other factors
    # rhyme count ratio
    r_ratio = 0 if max_equ_slc_sum == 0 else rc_ing/max_equ_slc_sum
    # acmp_sr also effects extra score
    extscore = EXTRA_POINTS * acmp_sr
    if 0.6 <= r_ratio <= 0.8:
        # no deduction when in the best ratio range
        extscore *= 1.0
    elif pll_rc == rc_ino == rc_ing and pll_rc > 0:
        # or if it meet the requirement, reward on a scale of 0.7
        extscore *= 0.7
    elif (r_delta := abs(r_ratio-0.7)) <= 0.3:
        # or if it's close to middle of the best ratio range
        # (delta <= 0.3, [0.4,0.6), (0.8, 1.0]), reward on a scale
        # of 0.4*(1-delta)
        extscore *= 0.4 * (1-r_delta)
    else:
        extscore *= 0.0
    fnl_sum_score += extscore

    detail_record += '\nextra    score:{:>8.4f}, r_ratio={:.8f}'\
        ' (pll_rc={})'.format(extscore, r_ratio, pll_rc)
    detail_record += f'\n*FNL-SCORE-SUM: {fnl_sum_score}'

    als_diff_html = HtmlDiff(
        charjunk=None, tabsize=4, wrapcolumn=100
    ).make_file(
        fromlines=o_msstr.splitlines(keepends=True),
        tolines=g_msstr.splitlines(keepends=True),
        fromdesc='original mss',
        todesc='generated mss'
    )

    # scores = (fnl_sum_score, ph1score, p21score, p22score,\
    #           ph3score, ph4score, extscore)
    # return scores, detail_record, als_diff_html
    return fnl_sum_score, detail_record, als_diff_html



if __name__ == '__main__':
    o_msstr = '(verse)\ncccccccc\nccccccR\ncccccccc\nccccccR\n(verse)\ncccccccccR\nccccccccc\ncccccccccR\nccccccccc\n(chorus)\nccccccc\nccccccccR\nccccR\nccccccccR\n(chorus)\nccccccR\nccccccccR\nccccR\nccccccccc\n(verse)\ncccccccc\nccccccR\ncccccccc\nccccccR\n(verse)\ncccccccccR\nccccccccc\ncccccccccR\nccccccccc\n(chorus)\nccccccc\nccccccccR\nccccR\nccccccccR\n(chorus)\nccccccR\nccccccccR\nccccR\nccccccccc(verse)\ncccccccc\nccccccR\ncccccccc\nccccccR\n(verse)\ncccccccccR\nccccccccc\ncccccccccR\nccccccccc'

    gen_lrc = '(verse)\n如果一切可以重新来过\n我会在相遇的街口\n把你紧握再也不放手\n不会再让你远走\n(verse)\n如果上天能再给我一次\n爱你的机会不会辜负\n也许这次机会会更加\n更加珍惜不再错过你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每天都能见到你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每一分都为你而活\n(others)\n如果一切可以重新来过\n我会在相遇的街口\n把你紧握再也不放手\n不会再让你远走\n(verse)\n如果上天能再给我一次\n爱你的机会不会辜负\n也许这次机会会更加\n更加珍惜不再错过你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每天都能见到你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每一分都为你而活\n(verse)\n如果一切可以重新来过\n我会在相遇的街口\n把你紧握再也不放手\n不会再让你远走\n(verse)\n如果上天能再给我一次\n爱你的机会不会辜负\n也许这次机会会更加\n更加珍惜不再错过你'

    _, log, html = scoring(gen_lrc, o_msstr)
    print(log)
    # print(html)




    # o_msstr = '(verse)\ncccccccc\nccccccR\ncccccccc\nccccccR\n(verse)\ncccccccccR\nccccccccc\ncccccccccR\nccccccccc\n(chorus)\nccccccc\nccccccccR\nccccR\nccccccccR\n(chorus)\nccccccR\nccccccccR\nccccR\nccccccccc'

    # gen_lrc1002 = '(verse)\n如果一切可以重新来过\n我会在相遇的街口\n把你紧握再也不放手\n不会再让你远走\n(verse)\n如果上天能再给我一次\n爱你的机会不会辜负\n也许这次机会会更加\n更加珍惜不再错过你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每天都能见到你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每一分都为你而活'

    # gen_lrc2001 = '(verse)\n未能成为你的知己\n是我一生的遗憾\n等到理解了那一切\n已经消失的太远\n(verse)\n理想的爱情是一份浪漫\n现实的爱情不完全是\n理想的爱情就像满花园\n现实的爱情就是摘取\n(chorus)\n既然上天再给我\n一次爱你的机会啊哈\n每一段开始\n都会有有一句祝你生日快乐\n(chorus)\n既然上天再给我\n一次爱你的机会啊哈\n每一天开始\n都会有有一句祝你生日快乐'

    # gen_lrc3001 = '(verse)\n因为没勇气说不出\n也看不清自己\n总是让那份想念\n留在心里给你\n(verse)\n如果现在有台时光机器\n我会回去哪里遇见你\n不能再让你哭泣告诉你\n我会陪伴你到天亮里\n(chorus)\n每一次每一次\n每一份我总会想起你\n想到见到你\n心跳还是会加速不已\n(chorus)\n每一次每一次\n想见你你就是那么近\n想到见到你\n我会一直守护在这里'

    # gen_lrc3_1001 = '(verse)\n曾路过多少下雨天\n天空也曾雷鸣电闪\n当我擦干你眼角的泪\n心疼的感觉多严重影响\n(verse)\n曾拥有过多少海誓山盟\n也流过多少真情眼泪\n当我面对冰冷的雨水\n记忆的画面一幕一幕\n(chorus)\n假如能还有时间\n我会对你说些甜言蜜语\n不会让你走\n留下灿烂的烟花飞舞\n(chorus)\n假如还能有一天\n那是从你的眼前到心头\n每一刻低头\n都能看见你美丽的双眸'

    # gen_lrc3_2001 = '(verse)\n相遇的茫然相识的喜欢\n相知的 natural 歌名\n相爱的无奈逃不开的离别\n错过的都是那么的显眼\n(verse)\n我们需要时间去明白然后\n才能够懂得如何的去爱\n我相信这个世界没有改变\n还是每时每刻都会为\n(chorus)\n请你给我个机会\n让我再回到那年的开始\n好想对着你说\n每一刻都能为你而开始\n(chorus)\n就算是会有不开心\n就算会有千分万分的伤心\n我都不会有半点后悔\n相信这个世界没有事不能够'

    # print(f'FS of gen_lrc1002  : {scoring(gen_lrc1002  , o_msstr)}\n')
    # print(f'FS of gen_lrc2001  : {scoring(gen_lrc2001  , o_msstr)}\n')
    # print(f'FS of gen_lrc3001  : {scoring(gen_lrc3001  , o_msstr)}\n')
    # print(f'FS of gen_lrc3_1001: {scoring(gen_lrc3_1001, o_msstr)}\n')
    # print(f'FS of gen_lrc3_2001: {scoring(gen_lrc3_2001, o_msstr)}\n')


    # o_msstr = '(verse1)\nccccccccccR\ncccccccR\ncccccccccccR\ncccccccR\n(chorus1)\ncccccccccccccR\nccccccccccR\n(chorus2)\ncccccccccccccR\nccccccccccR\n(verse2)\ncccccccccR\ncccccccR\ncccccccccccR\ncccccccc\n(chorus1)\ncccccccccccccR\nccccccccccR\n(chorus2)\ncccccccccccccR\nccccccccccR\n(bridge1)\nccccccccccccR\ncccccccR\n(chorus1)\ncccccccccccccR\nccccccccccR\n(chorus3)\ncccccccccccc\nccccccccccc\n(ending1)\ncccccccccccR\ncccccccR'
    # # 原始歌词
    # gen_lrc = '(verse1)\n让风告诉你我不会想太多\n淡淡的歌简单的我\n孤单的人不多只剩下一个我\n如今我已不再是我\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus2)\n也许落叶的时候或许飘雪的时候\n心里的你才会陪我一起走\n(verse2)\n你离开我连风声都沉默\n快乐已经不属于我\n痴心的人不多只剩下一个我\n如今我已懂得执着\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus2)\n也许落叶的时候或许飘雪的时候\n心里的你才会陪我一起走\n(bridge1)\n我一直在等待不计较你的伤害\n期待着你能再回来\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus3)\n也许落叶的时候也只是也许\n心里的你才会陪我一起走\n(ending1)\n还剩下些什么一个落叶的我\n笑着流泪这就是我'
    # # 少了一行
    # gen_lrc1 = '(verse1)\n让风告诉你我不会想太多\n淡淡的歌简单的我\n孤单的人不多只剩下一个我\n如今我已不再是我\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus2)\n也许落叶的时候或许飘雪的时候\n(verse2)\n你离开我连风声都沉默\n快乐已经不属于我\n痴心的人不多只剩下一个我\n如今我已懂得执着\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus2)\n也许落叶的时候或许飘雪的时候\n心里的你才会陪我一起走\n(bridge1)\n我一直在等待不计较你的伤害\n期待着你能再回来\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus3)\n也许落叶的时候也只是也许\n心里的你才会陪我一起走\n(ending1)\n还剩下些什么一个落叶的我\n笑着流泪这就是我'
    # # 改一段名
    # gen_lrc2 = '(verse1)\n让风告诉你我不会想太多\n淡淡的歌简单的我\n孤单的人不多只剩下一个我\n如今我已不再是我\n(verse)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus2)\n也许落叶的时候或许飘雪的时候\n心里的你才会陪我一起走\n(verse2)\n你离开我连风声都沉默\n快乐已经不属于我\n痴心的人不多只剩下一个我\n如今我已懂得执着\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus2)\n也许落叶的时候或许飘雪的时候\n心里的你才会陪我一起走\n(bridge1)\n我一直在等待不计较你的伤害\n期待着你能再回来\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus3)\n也许落叶的时候也只是也许\n心里的你才会陪我一起走\n(ending1)\n还剩下些什么一个落叶的我\n笑着流泪这就是我'
    # # 少个8字（8行各1个）
    # gen_lrc3 = '(verse1)\n让风告诉你我不会想多\n淡淡的歌简单的我\n孤单的人不多只剩下一个我\n如今我已不再是我\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus2)\n也许落叶的时候或飘雪的时候\n心里的你才会陪我一起走\n(verse2)\n你离开我连风声都沉默\n快乐已经不属于我\n痴心的人不多剩下一个我\n如今我已懂得执着\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时会有尽头\n(chorus2)\n也许落叶的时候或许飘雪的时候\n心里的你才会陪我一起走\n(bridge1)\n我一直在等不计较你的伤害\n期待着你能再回来\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时会有尽头\n(chorus3)\n也许落叶的时候也只是也许\n心里的你才会陪我一起走\n(ending1)\n还剩些什么一个落叶的我\n笑着流泪这是我'
    # # 多个5字（1行多5个）
    # gen_lrc4 = '(verse1)\n让风告诉你我不会想太多\n淡淡的歌简单的我\n孤单的人不多只剩下一个我\n如今我已不再是我\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus2)\n也许落叶的时候或AAAAAAAA许飘雪的时候\n心里的你才会陪我一起走\n(verse2)\n你离开我连风声都沉默\n快乐已经不属于我\n痴心的人不多只剩下一个我\n如今我已懂得执着\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus2)\n也许落叶的时候或许飘雪的时候\n心里的你才会陪我一起走\n(bridge1)\n我一直在等待不计较你的伤害\n期待着你能再回来\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus3)\n也许落叶的时候也只是也许\n心里的你才会陪我一起走\n(ending1)\n还剩下些什么一个落叶的我\n笑着流泪这就是我'
    # 改动了很多...
    # gen_lrc5 = '(verse1)\n让风告诉你我不会去想\n淡淡的歌简单的我\n孤单的人不多只剩下你我\n如今我已不再是那个少年\n(chorus2)\n也许是落叶或许是飘雪\n怎样才会陪我一起走\n(verse2)\n你离开连风声都沉默\n快乐已经不属于这里\n痴心的人不多只剩下一个我\n如今我已懂得执着\n(bridge)\n就在落叶的时候\n还是飘雪的时候\n我的寂寞何时才会有尽头\n(chorus2)\n也许落叶的时候或许飘雪的时候\n心里的你才会陪我一起走\n(bridge1)\n我一直在等待不计较你的伤害\n期待着你能再回来\n(chorus1)\n就在落叶的时候还是飘雪的时候\n我的寂寞何时才会有尽头\n(others)\n也许落叶的时候也只是也许\n心里的你才会陪我一起走\n(ending1)\n还剩下些什么一个落叶的我\n笑着流泪这就是我'

    # print(f'final score of gen_lrc : {scoring(gen_lrc , o_msstr)}\n')
    # print(f'final score of gen_lrc1: {scoring(gen_lrc1, o_msstr)}\n')
    # print(f'final score of gen_lrc2: {scoring(gen_lrc2, o_msstr)}\n')
    # print(f'final score of gen_lrc3: {scoring(gen_lrc3, o_msstr)}\n')
    # print(f'final score of gen_lrc4: {scoring(gen_lrc4, o_msstr)}\n')
    # print(f'final score of gen_lrc5: {scoring(gen_lrc5, o_msstr)}\n')



    # from difflib import HtmlDiff
    # a = [
    #     '(verse)\n',
    #     'ccccR\n',
    #     'ccccR\n',
    #     'cccccR\n',
    #     '(chorus)\n',
    #     'ccccccc cR\n',
    #     'ccccccc cR\n',
    #     'ccccccccc\n',
    #     'ccccccccR',
    # ]
    # b = [
    #     '(verse)',
    #     'ccccR\n',
    #     'ccccc\n',
    #     'cccccR\n',
    #     '(chorus)',
    #     'ccccccc cR\n',
    #     'cccccc ccR\n',
    #     'ccccccccc\n',
    # ]
    # html = HtmlDiff(charjunk=None).make_file(a, b, 'from a', 'to b')
    # with open('./ab_diff.html', "w", encoding="utf-8") as file:
    #     file.write(html)